#pragma once
// adapter-merge.h: runtime adapter merge into GGUF weights before QKV fusion.
//
// Supported algorithms:
//   LoRA (low rank adaptation):  delta = (alpha / rank) * B @ A
//   LoKr (low rank Kronecker):   delta = (alpha / rank) * kron(w1, w2_a @ w2_b)
//                                optional DoRA magnitude rescale per output row
//                                when a dora_scale tensor is present.
//
// Called after individual GGUF projection tensors are loaded into WeightCtx
// but BEFORE wctx_alloc uploads to GPU and BEFORE QKV fusion concatenation.
//
// Each projection (q_proj, k_proj, v_proj, o_proj) has its own PendingCopy
// even when destined for a fused QKV tensor. We patch each one separately,
// so fusion proceeds normally on already merged data.
//
// Performance: both paths dispatch the expensive dense math to the best
// available backend via ggml_backend_graph_compute. LoRA runs one scaled
// B @ A per tensor. LoKr runs one kron(w1, w2_a @ w2_b) per tensor. The
// backend graph is strictly the dense compute; BF16 rounding, optional
// DoRA rescale, base add, and requantization all happen on host row by
// row, so no full F32 base is ever allocated or uploaded. On CUDA the
// backend uses cuBLAS plus elementwise kernels, on CPU it uses ggml's
// threaded SIMD kernels, Vulkan and Metal get their respective backends.
// PendingCopy lookup is O(1) via hashmap.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "safetensors.h"
#include "weight-ctx.h"

#include <sys/stat.h>
#ifdef _WIN32
#    ifndef S_ISDIR
#        define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#    endif
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Convert safetensors tensor data to F32 based on dtype string.
// Handles "F32", "BF16", "F16". Returns false for unknown dtypes.
static bool adapter_to_f32(const void * src, float * dst, int64_t n, const std::string & dtype) {
    if (dtype == "F32") {
        memcpy(dst, src, (size_t) n * sizeof(float));
    } else if (dtype == "BF16") {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
    } else {
        return false;
    }
    return true;
}

// Map a LoRA safetensors key to the GGUF base tensor name.
//
// Supported key formats (all map to GGUF "decoder.layers.0.self_attn.q_proj.weight"):
//
//   PEFT adapter_model.safetensors:
//     base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
//     base_model.model.layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (no prefix):
//     layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (diffusion_model prefix):
//     diffusion_model.layers.0.self_attn.q_proj.lora_A.weight
//
// Steps: strip known prefix, extract module path before ".lora_",
// prepend "decoder." if needed, append ".weight".
static std::string lora_base_name(const std::string & key) {
    std::string s = key;

    // strip known prefixes (PEFT, ComfyUI)
    static const char * prefixes[] = {
        "base_model.model.",  // PEFT
        "diffusion_model.",   // ComfyUI official ACE-Step format
    };
    for (const char * pfx : prefixes) {
        size_t pfx_len = strlen(pfx);
        if (s.compare(0, pfx_len, pfx) == 0) {
            s = s.substr(pfx_len);
            break;
        }
    }

    // everything before ".lora_" is the module path
    size_t pos = s.find(".lora_");
    if (pos == std::string::npos) {
        return "";
    }
    s = s.substr(0, pos);

    // ensure decoder prefix (PEFT wraps the decoder directly,
    // so the internal path starts at "layers." not "decoder.layers.")
    if (s.compare(0, 8, "decoder.") != 0) {
        s = "decoder." + s;
    }

    return s + ".weight";
}

// Check whether a safetensors key is a lora_A/down or lora_B/up weight.
// PEFT uses .lora_A. / .lora_B., ComfyUI single-file uses .lora_down. / .lora_up.
static bool lora_is_a(const std::string & key) {
    return key.find(".lora_A.") != std::string::npos || key.find(".lora_down.") != std::string::npos;
}

static bool lora_is_b(const std::string & key) {
    return key.find(".lora_B.") != std::string::npos || key.find(".lora_up.") != std::string::npos;
}

// Read adapter_config.json for alpha. Returns alpha or 0 if not found.
// Rank is always read from the actual tensor shapes (more reliable).
static int adapter_read_alpha(const char * dir) {
    std::string path = std::string(dir) + "/adapter_config.json";

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<char> buf((size_t) len + 1);
    size_t            nr = fread(buf.data(), 1, (size_t) len, f);
    fclose(f);
    if (nr != (size_t) len) {
        return 0;
    }
    buf[(size_t) len] = '\0';

    const char * json  = buf.data();
    int          alpha = 0;

    // look for "lora_alpha": <int>
    const char * p = strstr(json, "\"lora_alpha\"");
    if (p) {
        p = strchr(p + 12, ':');
        if (p) {
            alpha = atoi(p + 1);
        }
    }

    // fallback: try "alpha": <int> (some configs use this)
    if (alpha == 0) {
        p = strstr(json, "\"alpha\"");
        if (p) {
            p = strchr(p + 7, ':');
            if (p) {
                alpha = atoi(p + 1);
            }
        }
    }

    if (alpha > 0) {
        fprintf(stderr, "[Adapter] adapter_config.json: alpha=%d\n", alpha);
    }
    return alpha;
}

// Dequant a GGUF tensor buffer to F32 using ggml type traits.
// Works for all types: F32, BF16, F16, Q4_K, Q8_0, etc.
static void adapter_dequant(const void * src, float * dst, int64_t nel, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) nel * sizeof(float));
        return;
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits->to_float) {
        traits->to_float(src, dst, nel);
    } else {
        fprintf(stderr, "[Adapter] WARNING: no dequant for type %d, zeroing\n", type);
        memset(dst, 0, (size_t) nel * sizeof(float));
    }
}

// Requant F32 data back to original type. Writes into dst buffer.
// Returns the number of bytes written.
static size_t adapter_requant(const float * src, void * dst, int64_t nel, int64_t n_per_row, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        size_t nb = (size_t) nel * sizeof(float);
        memcpy(dst, src, nb);
        return nb;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(type);

    if (traits->is_quantized) {
        // quantized types: use ggml_quantize_chunk (handles block alignment)
        int64_t nrows = nel / n_per_row;
        size_t  qsize = ggml_row_size(type, n_per_row) * (size_t) nrows;
        ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);
        return qsize;
    }

    // non quantized (BF16, F16): use from_float_ref
    if (traits->from_float_ref) {
        size_t nb = (size_t) nel * traits->type_size;
        traits->from_float_ref(src, dst, nel);
        return nb;
    }

    fprintf(stderr, "[Adapter] WARNING: no requant for type %d\n", type);
    return 0;
}

// Round F32 data through BF16 in place to match PEFT's intermediate precision.
// Processes in fixed chunks to avoid large stack allocations.
static void adapter_bf16_round(float * data, int64_t n) {
    const int64_t chunk = 4096;
    ggml_bf16_t   tmp[4096];
    for (int64_t i = 0; i < n; i += chunk) {
        int64_t len = (n - i < chunk) ? n - i : chunk;
        ggml_fp32_to_bf16_row_ref(data + i, tmp, len);
        ggml_bf16_to_fp32_row(tmp, data + i, len);
    }
}

// Compute delta = scaling * B @ A via the best available backend.
//
// On CUDA: tensors are uploaded to GPU, cuBLAS does the GEMM, result
// is downloaded back. On CPU: ggml's threaded SIMD kernels run locally.
// The backend abstraction handles all memory management transparently.
//
// A is [rank, in_feat] row-major (safetensors convention).
// B is [out_feat, rank] row-major.
// Writes out_feat * in_feat floats into delta.
static void adapter_gemm(const float *  a,
                         const float *  b,
                         float *        delta,
                         int64_t        out_feat,
                         int64_t        rank,
                         int64_t        in_feat,
                         float          scaling,
                         ggml_backend_t backend) {
    int64_t a_nel = rank * in_feat;
    int64_t b_nel = out_feat * rank;
    int64_t c_nel = out_feat * in_feat;

    // metadata context: tensor descriptors + graph only, no data
    // (the backend allocates tensor memory via ggml_backend_alloc_ctx_tensors)
    size_t                  meta   = ggml_tensor_overhead() * 6 + ggml_graph_overhead() + 4096;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);

    // input tensors store raw row-major data from safetensors.
    // ggml is column-major, so row-major A[rank, in_feat] maps to ggml [in_feat, rank].
    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, rank);
    struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_feat);

    // graph: transpose A so rank becomes ne[0] (contraction dim), then GEMM + scale.
    // ggml_cont materializes the transposed view into a contiguous tensor.
    // on CUDA, all three ops run as GPU kernels (transpose copy + cuBLAS + scale).
    struct ggml_tensor * ta_t   = ggml_cont(ctx, ggml_transpose(ctx, ta));
    struct ggml_tensor * result = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb), scaling);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // allocate all tensor data on the backend (GPU VRAM or CPU RAM)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // upload: host to device (noop when backend is CPU)
    ggml_backend_tensor_set(ta, a, 0, (size_t) a_nel * sizeof(float));
    ggml_backend_tensor_set(tb, b, 0, (size_t) b_nel * sizeof(float));

    // compute: transpose + matmul + scale, all on the backend
    ggml_backend_graph_compute(backend, graph);

    // download: device to host (noop when backend is CPU)
    ggml_backend_tensor_get(result, delta, 0, (size_t) c_nel * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Build the reverse map from LyCORIS key prefix to GGUF tensor name.
// LyCORIS stores adapter tensors as "lycoris_<path_with_underscores>.<suffix>",
// where the torch module path has all dots flattened to underscores. We cannot
// safely reverse that transform blindly (names like "cross_attn" contain real
// underscores), so we enumerate the GGUF decoder .weight tensors we already
// have and build the mapping from them.
//
// Example:
//   GGUF tensor "decoder.layers.0.self_attn.q_proj.weight"
//   -> "lycoris_layers_0_self_attn_q_proj" -> "decoder.layers.0.self_attn.q_proj.weight"
static std::unordered_map<std::string, std::string> lokr_build_reverse_map(const GGUFModel & gf) {
    std::unordered_map<std::string, std::string> out;
    int                                          n_tensors = (int) gguf_get_n_tensors(gf.gguf);
    static const char *                          suffix    = ".weight";
    size_t                                       slen      = strlen(suffix);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gf.gguf, i);
        std::string  s    = name;
        // only decoder.*.weight tensors qualify as LoKr targets
        if (s.size() <= slen || s.compare(s.size() - slen, slen, suffix) != 0) {
            continue;
        }
        if (s.compare(0, 8, "decoder.") != 0) {
            continue;
        }
        // strip "decoder." prefix and ".weight" suffix, flatten dots to underscores
        std::string path = s.substr(8, s.size() - 8 - slen);
        for (char & c : path) {
            if (c == '.') {
                c = '_';
            }
        }
        out["lycoris_" + path] = s;
    }
    return out;
}

// Split a safetensors key on its last dot. Returns false when no dot exists.
// Example: "lycoris_layers_0_self_attn_q_proj.lokr_w1"
//   -> prefix "lycoris_layers_0_self_attn_q_proj", suffix "lokr_w1"
static bool adapter_split_suffix(const std::string & key, std::string * prefix, std::string * suffix) {
    size_t dot = key.rfind('.');
    if (dot == std::string::npos) {
        return false;
    }
    *prefix = key.substr(0, dot);
    *suffix = key.substr(dot + 1);
    return true;
}

// Compute LoKr delta = (alpha / rank) * kron(w1, w2_a @ w2_b) on the best
// available backend. Architecturally mirrors adapter_gemm for LoRA: the
// backend runs only the expensive kron math, the caller handles BF16
// rounding, optional DoRA rescale, base add, and requantization on host
// row by row. Keeps the graph minimal, avoids any F32 base allocation or
// upload, and cuts PCIe traffic to the three tiny adapter factors plus
// the final delta download.
//
// Shapes (host row major -> ggml ne):
//   w1   (a, b)        -> ne=(b, a)
//   w2_a (c, r)        -> ne=(r, c)
//   w2_b (r, d)        -> ne=(d, r)
//   delta (a*c, b*d)   -> ne=(b*d, a*c)
//
// Graph:
//   tw2b_T = cont(transpose(tw2b))                         ne=(r, d)
//   tw2    = mul_mat(tw2b_T, tw2a)                         ne=(d, c)
//   tw1_s  = scale(tw1, alpha / rank)                      ne=(b, a), tiny
//   touter = mul_mat(reshape(tw1_s, 1, a*b),
//                    reshape(tw2, 1, c*d))                 ne=(a*b, c*d)
//   kron_p = permute(reshape_4d(touter, b,a,d,c), 1,3,0,2) ne=(d, b, c, a)
//   delta  = reshape_2d(cont(kron_p), b*d, a*c)
//
// ggml_permute axis_i positions src axis i AT new ne index axis_i
// (ggml.c:3781 sets ne[axis_i] = src.ne[i]), so mapping src (b, a, d, c)
// -> new (d, b, c, a) needs src axes (0, 1, 2, 3) to land at new positions
// (1, 3, 0, 2). The fast pair (d, b) then collapses into in_feat and the
// slow pair (c, a) into out_feat under reshape_2d. Net effect:
//   delta_rm[aa*c + cc, bb*d + dd] = W1[aa, bb] * W2[cc, dd]
//
// The kron cannot use a straight ggml_mul broadcast: ggml_mul requires one
// side to already have the full broadcast shape, and neither (1, b, 1, a)
// nor (d, 1, c, 1) covers (b*d, a*c) alone. An outer product via mul_mat
// always materializes the full (a*b, c*d) tensor, after which one axis
// swap yields the kron layout.
static void lokr_kron_gemm(const float *  w1,
                           int64_t        a,
                           int64_t        b,
                           const float *  w2_a,
                           int64_t        c,
                           int64_t        r,
                           const float *  w2_b,
                           int64_t        d,
                           float          scaling,
                           float *        delta,
                           ggml_backend_t backend) {
    int64_t in_feat  = b * d;
    int64_t out_feat = a * c;
    int64_t nel      = in_feat * out_feat;

    // metadata context: input tensors + intermediate nodes + graph + slack
    size_t                  meta   = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 4096;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);

    struct ggml_tensor * tw1  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, b, a);
    struct ggml_tensor * tw2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r, c);
    struct ggml_tensor * tw2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, r);

    // w2 = w2_a @ w2_b: transpose w2_b so rank sits on ne[0] (contraction dim)
    struct ggml_tensor * tw2b_T = ggml_cont(ctx, ggml_transpose(ctx, tw2b));
    struct ggml_tensor * tw2    = ggml_mul_mat(ctx, tw2b_T, tw2a);

    // scale alpha / rank on the tiny (typically 4x4) side, cheapest placement
    struct ggml_tensor * tw1_s = ggml_scale(ctx, tw1, scaling);

    // kron via outer product + axis swap, see header comment
    struct ggml_tensor * tw1_flat  = ggml_reshape_2d(ctx, tw1_s, 1, a * b);
    struct ggml_tensor * tw2_flat  = ggml_reshape_2d(ctx, tw2, 1, c * d);
    struct ggml_tensor * touter    = ggml_mul_mat(ctx, tw1_flat, tw2_flat);
    struct ggml_tensor * touter_4d = ggml_reshape_4d(ctx, touter, b, a, d, c);
    struct ggml_tensor * tkron_p   = ggml_permute(ctx, touter_4d, 1, 3, 0, 2);
    struct ggml_tensor * tkron_c   = ggml_cont(ctx, tkron_p);
    struct ggml_tensor * tdelta    = ggml_reshape_2d(ctx, tkron_c, in_feat, out_feat);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, tdelta);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(tw1, w1, 0, (size_t) (a * b) * sizeof(float));
    ggml_backend_tensor_set(tw2a, w2_a, 0, (size_t) (c * r) * sizeof(float));
    ggml_backend_tensor_set(tw2b, w2_b, 0, (size_t) (r * d) * sizeof(float));

    ggml_backend_graph_compute(backend, graph);

    ggml_backend_tensor_get(tdelta, delta, 0, (size_t) nel * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Detect whether the safetensors payload is a LyCORIS LoKr pack.
// Any tensor named "*.lokr_w1" or "*.lokr_w2*" is a LoKr marker that LoRA
// payloads never produce, so a single match is sufficient.
static bool adapter_detect_lokr(const STFile & st) {
    for (const auto & e : st.entries) {
        if (e.name.find(".lokr_w1") != std::string::npos || e.name.find(".lokr_w2") != std::string::npos) {
            return true;
        }
    }
    return false;
}

// Find the PendingCopy whose src pointer matches the GGUF mmap location of a
// tensor, patch it with a newly allocated staging buffer containing the merged
// weight, and return true on success. The F32 merged data in `merged_f32` is
// quantized back to the tensor's native GGUF type in place into wctx->staging.
static bool adapter_patch_pending(WeightCtx *                                wctx,
                                  std::unordered_map<const void *, size_t> & pending_idx,
                                  const void *                               base_ptr,
                                  float *                                    merged_f32,
                                  int64_t                                    nel,
                                  int64_t                                    ne0,
                                  enum ggml_type                             ttype,
                                  const char *                               gguf_name) {
    auto pc_it = pending_idx.find(base_ptr);
    if (pc_it == pending_idx.end()) {
        fprintf(stderr, "[Adapter] WARNING: no PendingCopy for %s, skipping\n", gguf_name);
        return false;
    }
    WeightCtx::PendingCopy * pc = &wctx->pending[pc_it->second];

    size_t max_bytes = (size_t) nel * sizeof(float);
    size_t n_floats  = (max_bytes + sizeof(float) - 1) / sizeof(float);
    wctx->staging.emplace_back(n_floats);
    void * merged_buf = wctx->staging.back().data();

    size_t merged_bytes = adapter_requant(merged_f32, merged_buf, nel, ne0, ttype);
    if (merged_bytes == 0) {
        return false;
    }

    pc->src    = merged_buf;
    pc->nbytes = merged_bytes;
    return true;
}

// Add base weight to delta row by row. Each row is dequantized into a small
// scratch buffer, accumulated into delta, then discarded. Avoids a second
// full size F32 allocation.
static void adapter_add_base(const void * base_ptr, enum ggml_type ttype, int64_t ne0, int64_t ne1, float * delta) {
    size_t             row_bytes = ggml_row_size(ttype, ne0);
    std::vector<float> row_buf((size_t) ne0);
    for (int64_t i = 0; i < ne1; i++) {
        const void * row_src   = (const uint8_t *) base_ptr + i * row_bytes;
        float *      delta_row = delta + i * ne0;
        adapter_dequant(row_src, row_buf.data(), ne0, ttype);
        for (int64_t j = 0; j < ne0; j++) {
            delta_row[j] += row_buf[j];
        }
    }
}

// Merge base into delta with DoRA magnitude decomposition per output row.
// Matches LyCORIS apply_weight_decompose at lycoris/modules/lokr.py for non
// bypass mode. Base is dequantized one row at a time into a scratch buffer
// so no full F32 base allocation is needed.
//
//   m     = base + delta                                    per row
//   norm  = sqrt(sum(m ^ 2)) + eps                          per row
//   scale = user_scale * (dora_scale / norm) + (1 - user_scale)
//   out   = m * scale                                       per row broadcast
//
// delta_inout starts as the scaled kron delta, ends as the full merged weight
// F32 ready for requantization. One pass per row: dequant, accumulate, sqsum,
// rescale. Host scalar cost is trivial next to the backend kron compute.
static void adapter_apply_dora(const void *   base_ptr,
                               enum ggml_type ttype,
                               int64_t        ne0,
                               int64_t        ne1,
                               const float *  ds,
                               float          user_scale,
                               float *        delta_inout) {
    // torch.finfo(torch.bfloat16).eps, used verbatim in LyCORIS
    const float eps = 7.8125e-3f;

    size_t             row_bytes = ggml_row_size(ttype, ne0);
    std::vector<float> row_buf((size_t) ne0);
    for (int64_t i = 0; i < ne1; i++) {
        const void * row_src = (const uint8_t *) base_ptr + i * row_bytes;
        float *      row_out = delta_inout + i * ne0;
        adapter_dequant(row_src, row_buf.data(), ne0, ttype);
        float sq_sum = 0.0f;
        for (int64_t j = 0; j < ne0; j++) {
            row_out[j] += row_buf[j];
            sq_sum += row_out[j] * row_out[j];
        }
        float norm  = std::sqrt(sq_sum) + eps;
        float scale = user_scale * (ds[i] / norm) + (1.0f - user_scale);
        for (int64_t j = 0; j < ne0; j++) {
            row_out[j] *= scale;
        }
    }
}

// LoRA merge path. Matches PEFT merge_and_unload for PEFT payloads and ComfyUI
// merge semantics for single file LoRA:
//   delta = (alpha / rank) * scale * B @ A
// Applied to base weights in place. Alpha is read per tensor if present
// (ComfyUI baked), else from adapter_config.json, else defaults to rank.
static bool adapter_merge_lora(WeightCtx *         wctx,
                               const GGUFModel &   gf,
                               const STFile &      st,
                               const std::string & cfg_dir,
                               float               scale,
                               ggml_backend_t      backend) {
    int alpha_cfg = adapter_read_alpha(cfg_dir.c_str());

    // group lora_A and lora_B entries by their GGUF base tensor name.
    // also collect per tensor alpha scalars (ComfyUI baked format).
    std::map<std::string, const STEntry *> a_map, b_map;
    std::map<std::string, float>           alpha_map;
    for (const auto & e : st.entries) {
        // per tensor alpha: "base_model.model.layers.0.self_attn.q_proj.alpha"
        // scalar F32 with shape [] containing the baked alpha value
        const char * alpha_suffix = ".alpha";
        size_t       slen         = strlen(alpha_suffix);
        if (e.name.size() > slen && e.name.compare(e.name.size() - slen, slen, alpha_suffix) == 0 && e.dtype == "F32" &&
            e.n_dims == 0) {
            std::string fake_key = e.name.substr(0, e.name.size() - slen) + ".lora_.x";
            std::string base     = lora_base_name(fake_key);
            if (!base.empty()) {
                float val = 0.0f;
                memcpy(&val, st_data(st, e), sizeof(float));
                alpha_map[base] = val;
            }
            continue;
        }

        std::string base = lora_base_name(e.name);
        if (base.empty()) {
            continue;
        }
        if (lora_is_a(e.name)) {
            a_map[base] = &e;
        } else if (lora_is_b(e.name)) {
            b_map[base] = &e;
        }
    }

    std::unordered_map<const void *, size_t> pending_idx;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
    }

    int merged  = 0;
    int skipped = 0;

    for (const auto & kv : a_map) {
        const std::string & gguf_name = kv.first;
        const STEntry *     ea        = kv.second;

        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) {
            fprintf(stderr, "[Adapter] WARNING: no lora_B for %s, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        const STEntry * eb = it->second;

        int64_t tidx = gguf_find_tensor(gf.gguf, gguf_name.c_str());
        if (tidx < 0) {
            fprintf(stderr, "[Adapter] WARNING: tensor %s not in GGUF, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, gguf_name.c_str());
        enum ggml_type       ttype = tmeta->type;
        int64_t              ne0   = tmeta->ne[0];
        int64_t              ne1   = tmeta->ne[1];
        int64_t              nel   = ne0 * ne1;

        size_t       toff     = gguf_get_tensor_offset(gf.gguf, tidx);
        const void * base_ptr = gf.mapping + gf.data_offset + toff;

        // LoRA shapes (safetensors PyTorch convention, row major):
        //   A: [rank, in_features]
        //   B: [out_features, rank]
        int64_t rank     = ea->shape[0];
        int64_t in_feat  = ea->shape[1];
        int64_t out_feat = eb->shape[0];

        if (eb->shape[1] != rank) {
            fprintf(stderr, "[Adapter] WARNING: rank mismatch A=%lld vs B=%lld for %s\n", (long long) rank,
                    (long long) eb->shape[1], gguf_name.c_str());
            skipped++;
            continue;
        }
        if (in_feat != ne0 || out_feat != ne1) {
            fprintf(stderr, "[Adapter] WARNING: shape mismatch for %s: LoRA [%lld,%lld] vs GGUF [%lld,%lld]\n",
                    gguf_name.c_str(), (long long) out_feat, (long long) in_feat, (long long) ne1, (long long) ne0);
            skipped++;
            continue;
        }

        // alpha: prefer per tensor (ComfyUI baked), then config, fallback to rank
        float alpha;
        auto  alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end()) {
            alpha = alpha_it->second;
        } else if (alpha_cfg > 0) {
            alpha = (float) alpha_cfg;
        } else {
            alpha = (float) rank;
        }
        float scaling = (alpha / (float) rank) * scale;

        int64_t            a_nel = rank * in_feat;
        int64_t            b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t) a_nel);
        std::vector<float> b_f32((size_t) b_nel);

        if (!adapter_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lora_A\n", ea->dtype.c_str());
            skipped++;
            continue;
        }
        if (!adapter_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype %s for lora_B\n", eb->dtype.c_str());
            skipped++;
            continue;
        }

        // PEFT casts LoRA weights to BF16 before computing the delta.
        // We replicate this round trip so B @ A matches merge_and_unload exactly.
        adapter_bf16_round(a_f32.data(), a_nel);
        adapter_bf16_round(b_f32.data(), b_nel);

        // delta = scaling * B @ A via backend GEMM (cuBLAS on CUDA, SIMD on CPU)
        std::vector<float> delta((size_t) nel);
        adapter_gemm(a_f32.data(), b_f32.data(), delta.data(), out_feat, rank, in_feat, scaling, backend);

        // round delta through BF16 to match PEFT intermediate precision.
        // without this, the diffusion model diverges (very weight sensitive).
        adapter_bf16_round(delta.data(), nel);

        adapter_add_base(base_ptr, ttype, ne0, ne1, delta.data());

        if (!adapter_patch_pending(wctx, pending_idx, base_ptr, delta.data(), nel, ne0, ttype, gguf_name.c_str())) {
            skipped++;
            continue;
        }
        merged++;
    }

    fprintf(stderr, "[Adapter] LoRA merged %d pairs (skipped %d), scale=%.2f\n", merged, skipped, scale);
    return merged > 0;
}

// LoKr merge path. Matches the LyCORIS runtime forward for LoKr at
// lycoris/modules/lokr.py:551..566 (non bypass mode, scalar=1):
//   delta       = (alpha / rank) * kron(w1, w2_a @ w2_b)
//   no DoRA     : merged = base + delta * multiplier
//   DoRA present: merged = apply_weight_decompose(base + delta, multiplier)
//
// Only the dense w2 factorization is supported (w2_a @ w2_b), which is what
// ACE-Step LoKr adapters use. w1 is expected whole, not factorized (no w1_a).
// Tucker decomposition and convolutional LoKr are not implemented.
static bool adapter_merge_lokr(WeightCtx *       wctx,
                               const GGUFModel & gf,
                               const STFile &    st,
                               float             user_scale,
                               ggml_backend_t    backend) {
    // group the five tensors per LyCORIS module prefix
    struct LoKrEntry {
        const STEntry * w1         = nullptr;
        const STEntry * w2_a       = nullptr;
        const STEntry * w2_b       = nullptr;
        const STEntry * alpha      = nullptr;
        const STEntry * dora_scale = nullptr;
    };

    std::map<std::string, LoKrEntry> modules;

    for (const auto & e : st.entries) {
        std::string prefix, suffix;
        if (!adapter_split_suffix(e.name, &prefix, &suffix)) {
            continue;
        }
        if (prefix.compare(0, 8, "lycoris_") != 0) {
            continue;
        }
        LoKrEntry & m = modules[prefix];
        if (suffix == "lokr_w1") {
            m.w1 = &e;
        } else if (suffix == "lokr_w2_a") {
            m.w2_a = &e;
        } else if (suffix == "lokr_w2_b") {
            m.w2_b = &e;
        } else if (suffix == "alpha") {
            m.alpha = &e;
        } else if (suffix == "dora_scale") {
            m.dora_scale = &e;
        }
    }

    std::unordered_map<std::string, std::string> name_map = lokr_build_reverse_map(gf);

    std::unordered_map<const void *, size_t> pending_idx;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
    }

    int merged     = 0;
    int skipped    = 0;
    int dora_count = 0;

    for (const auto & kv : modules) {
        const std::string & lyc_prefix = kv.first;
        const LoKrEntry &   m          = kv.second;

        if (!m.w1 || !m.w2_a || !m.w2_b || !m.alpha) {
            fprintf(stderr, "[Adapter] WARNING: incomplete LoKr module %s, skipping\n", lyc_prefix.c_str());
            skipped++;
            continue;
        }

        auto nm_it = name_map.find(lyc_prefix);
        if (nm_it == name_map.end()) {
            fprintf(stderr, "[Adapter] WARNING: no GGUF tensor for %s, skipping\n", lyc_prefix.c_str());
            skipped++;
            continue;
        }
        const std::string & gguf_name = nm_it->second;

        int64_t tidx = gguf_find_tensor(gf.gguf, gguf_name.c_str());
        if (tidx < 0) {
            fprintf(stderr, "[Adapter] WARNING: tensor %s not in GGUF, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, gguf_name.c_str());
        enum ggml_type       ttype = tmeta->type;
        int64_t              ne0   = tmeta->ne[0];
        int64_t              ne1   = tmeta->ne[1];
        int64_t              nel   = ne0 * ne1;

        size_t       toff     = gguf_get_tensor_offset(gf.gguf, tidx);
        const void * base_ptr = gf.mapping + gf.data_offset + toff;

        // LoKr shapes (safetensors row major):
        //   w1   : (a, b)                      with a = b = 4 for factor=4
        //   w2_a : (c, rank)
        //   w2_b : (rank, d)
        // Kronecker product yields (a*c, b*d) = (out_feat, in_feat) = (ne1, ne0).
        int64_t a  = m.w1->shape[0];
        int64_t b  = m.w1->shape[1];
        int64_t c  = m.w2_a->shape[0];
        int64_t r  = m.w2_a->shape[1];
        int64_t r2 = m.w2_b->shape[0];
        int64_t d  = m.w2_b->shape[1];

        if (r != r2) {
            fprintf(stderr, "[Adapter] WARNING: LoKr rank mismatch w2_a=%lld vs w2_b=%lld for %s\n", (long long) r,
                    (long long) r2, lyc_prefix.c_str());
            skipped++;
            continue;
        }
        if (a * c != ne1 || b * d != ne0) {
            fprintf(stderr,
                    "[Adapter] WARNING: LoKr shape mismatch for %s: kron(%lldx%lld, %lldx%lld) = %lldx%lld vs GGUF "
                    "out=%lld in=%lld\n",
                    gguf_name.c_str(), (long long) a, (long long) b, (long long) c, (long long) d, (long long) (a * c),
                    (long long) (b * d), (long long) ne1, (long long) ne0);
            skipped++;
            continue;
        }

        // alpha scalar (shape []), dtype varies across trainers: F32, BF16, or F16
        float alpha = 0.0f;
        if (!adapter_to_f32(st_data(st, *m.alpha), &alpha, 1, m.alpha->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported alpha dtype %s for %s, skipping\n", m.alpha->dtype.c_str(),
                    lyc_prefix.c_str());
            skipped++;
            continue;
        }

        // load w1, w2_a, w2_b to F32
        int64_t            w1_nel  = a * b;
        int64_t            w2a_nel = c * r;
        int64_t            w2b_nel = r * d;
        std::vector<float> w1_f32((size_t) w1_nel);
        std::vector<float> w2a_f32((size_t) w2a_nel);
        std::vector<float> w2b_f32((size_t) w2b_nel);

        if (!adapter_to_f32(st_data(st, *m.w1), w1_f32.data(), w1_nel, m.w1->dtype) ||
            !adapter_to_f32(st_data(st, *m.w2_a), w2a_f32.data(), w2a_nel, m.w2_a->dtype) ||
            !adapter_to_f32(st_data(st, *m.w2_b), w2b_f32.data(), w2b_nel, m.w2_b->dtype)) {
            fprintf(stderr, "[Adapter] WARNING: unsupported dtype in LoKr module %s, skipping\n", lyc_prefix.c_str());
            skipped++;
            continue;
        }

        // dora_scale dequantized first when present, row count matched to ne1
        const float *      ds_ptr = nullptr;
        std::vector<float> ds_f32;
        if (m.dora_scale) {
            int64_t ds_out = m.dora_scale->shape[0];
            if (ds_out != ne1) {
                fprintf(stderr, "[Adapter] WARNING: dora_scale dim0 %lld != out_feat %lld for %s\n", (long long) ds_out,
                        (long long) ne1, gguf_name.c_str());
                skipped++;
                continue;
            }
            ds_f32.resize((size_t) ds_out);
            if (!adapter_to_f32(st_data(st, *m.dora_scale), ds_f32.data(), ds_out, m.dora_scale->dtype)) {
                fprintf(stderr, "[Adapter] WARNING: unsupported dora_scale dtype %s for %s\n",
                        m.dora_scale->dtype.c_str(), gguf_name.c_str());
                skipped++;
                continue;
            }
            ds_ptr = ds_f32.data();
        }

        // kron delta on backend, mirrors adapter_gemm for LoRA
        std::vector<float> delta((size_t) nel);
        float              scaling = alpha / (float) r;
        lokr_kron_gemm(w1_f32.data(), a, b, w2a_f32.data(), c, r, w2b_f32.data(), d, scaling, delta.data(), backend);

        // BF16 round mirrors LyCORIS diff.to(base.dtype) before the add
        adapter_bf16_round(delta.data(), nel);

        // merge base into delta per row: DoRA rescale when dora_scale is
        // present, plain scaled add otherwise. No full F32 base allocation.
        if (ds_ptr) {
            adapter_apply_dora(base_ptr, ttype, ne0, ne1, ds_ptr, user_scale, delta.data());
            dora_count++;
        } else {
            if (user_scale != 1.0f) {
                for (int64_t i = 0; i < nel; i++) {
                    delta[i] *= user_scale;
                }
            }
            adapter_add_base(base_ptr, ttype, ne0, ne1, delta.data());
        }

        if (!adapter_patch_pending(wctx, pending_idx, base_ptr, delta.data(), nel, ne0, ttype, gguf_name.c_str())) {
            skipped++;
            continue;
        }
        merged++;
    }

    fprintf(stderr, "[Adapter] LoKr merged %d modules (%d with DoRA, skipped %d), scale=%.2f\n", merged, dora_count,
            skipped, user_scale);
    return merged > 0;
}

// Main adapter merge entry point.
//
// Call after all GGUF tensors are loaded into wctx->pending but before wctx_alloc.
// Detects the adapter algorithm from the safetensors payload and dispatches to
// the matching merge path. The adapter_path can be a .safetensors file or a
// directory containing adapter_model.safetensors (PEFT LoRA), lokr_weights.safetensors
// (ACE-Step LoKr), or a single LyCORIS file.
static bool adapter_merge(WeightCtx *       wctx,
                          const GGUFModel & gf,
                          const char *      adapter_path,
                          float             scale,
                          ggml_backend_t    backend) {
    std::string sf_path = adapter_path;
    std::string dir     = adapter_path;

    struct stat sb;
    if (stat(adapter_path, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        // try PEFT layout first, then LyCORIS ACE-Step layout
        std::string peft_path = std::string(adapter_path) + "/adapter_model.safetensors";
        if (stat(peft_path.c_str(), &sb) == 0) {
            sf_path = peft_path;
        } else {
            sf_path = std::string(adapter_path) + "/lokr_weights.safetensors";
        }
    } else {
        size_t sep = dir.find_last_of("/\\");
        dir        = (sep != std::string::npos) ? dir.substr(0, sep) : ".";
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        return false;
    }

    bool ok;
    if (adapter_detect_lokr(st)) {
        ok = adapter_merge_lokr(wctx, gf, st, scale, backend);
    } else {
        ok = adapter_merge_lora(wctx, gf, st, dir, scale, backend);
    }

    st_close(&st);
    return ok;
}
