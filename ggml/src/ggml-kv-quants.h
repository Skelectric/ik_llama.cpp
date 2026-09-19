//
// Copyright (C) 2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

// Phase 4 (DeepSeek-V4.1) packed KV-cache storage types - CPU quantisers.
//
// Three storage-only types, one per (value grid, block size, scale dtype):
//
//   GGML_TYPE_FP4_B16_E4M3   e2m1 values, block 16, e4m3 scale  (compressed KV)
//   GGML_TYPE_FP4_B32_E8M0   e2m1 values, block 32, e8m0 scale  (indexer K and Q)
//   GGML_TYPE_FP8_B32_E8M0   e4m3 values, block 32, e8m0 scale  (window KV)
//
// The rule is the *reference* rule (dsv41-test/docs/phase4-fp4-kv-cache-plan.md
// section 2), not ik_llama's MXFP4 rule:
//
//   s = 2^ceil(log2(amax/6))   e2m1 / e8m0, floor 6 * 2^-126
//   s = e4m3(amax/6)           e2m1 / e4m3, floor 6 * 2^-9
//   s = 2^ceil(log2(amax/448)) e4m3 / e8m0, floor 1e-4
//
// then q = clamp(x/s, -max, max) rounded to the nearest grid value with ties
// to even. The rounding mode was verified against the reference's own CUDA
// kernels (dsv41-test/evidence/phase4/gpu-kernel-check.txt: RNE, 36/36
// configurations exact).
//
// Codes are packed 2 per byte in the *reference* order (element 2j in the low
// nibble, element 2j+1 in the high one - torch.float4_e2m1fn_x2), so a packed
// cache block can be compared byte-for-byte with the reference's own output.
//
// The types are storage-only: they are written by ggml_set_rows (from_float)
// and read back through ggml_get_rows -> F32 (to_float). They must never be a
// raw operand of a compute op (plan section 3.2, gate G7).

#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

void quantize_row_fp4_b16_e4m3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp4_b16_e4m3(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_fp4_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp4_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

void quantize_row_fp8_b32_e8m0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
void dequantize_row_fp8_b32_e8m0(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif