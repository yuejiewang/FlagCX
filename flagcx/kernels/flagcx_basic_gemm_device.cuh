#ifndef FLAGCX_BASIC_GEMM_DEVICE_CUH_
#define FLAGCX_BASIC_GEMM_DEVICE_CUH_

#include "device_utils.h"

FLAGCX_DEVICE_INLINE_DECORATOR void flagcxBasicGemmDevice(
    uint64_t a, uint64_t b, uint64_t c, uint64_t m, uint64_t n, uint64_t k,
    uint64_t lda, uint64_t ldb, uint64_t ldc, uint64_t nthreads,
    uint64_t accumulate) {
  const float *aPtr = reinterpret_cast<const float *>(a);
  const float *bPtr = reinterpret_cast<const float *>(b);
  float *cPtr = reinterpret_cast<float *>(c);
  uint64_t total = m * n;
  for (uint64_t linear = FLAGCX_THREAD_IDX_X; linear < total;
       linear += nthreads) {
    uint64_t row = linear / n;
    uint64_t col = linear % n;
    float sum = 0.0f;
    for (uint64_t p = 0; p < k; ++p) {
      sum += aPtr[row * lda + p] * bPtr[p * ldb + col];
    }
    uint64_t outputIdx = row * ldc + col;
    cPtr[outputIdx] = accumulate != 0 ? cPtr[outputIdx] + sum : sum;
  }
}

#endif // FLAGCX_BASIC_GEMM_DEVICE_CUH_
