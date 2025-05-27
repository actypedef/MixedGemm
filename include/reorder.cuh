#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include "cutlass/numeric_conversion.h"

typedef cutlass::float_e2m1_t fp4_t;
typedef cutlass::float_e3m2_t fp6_t;
typedef cutlass::float_e4m3_t fp8_t;
typedef cutlass::float_ue8m0_t sf_t;
typedef cutlass::bfloat16_t bf16_t;

template<int group_size, int hidden_dim>
void run_reorder_bf16_mixed(
  bf16_t *hidden_states,
  int seq_len,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
);

template<int group_size, int hidden_dim>
void run_reorder_bf16_fp4(
  bf16_t *hidden_states,
  int seq_len,
  int16_t *reorder_index,
  uint8_t *o_normal,
  uint8_t *o_sensitive,
  uint8_t *o_outlier,
  sf_t *normal_scale,
  sf_t *sensitive_scale,
  sf_t *outlier_scale,
  int KN, int KS, int KO
);