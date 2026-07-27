/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "kernel_operator.h"

#include <cstdint>

namespace {

constexpr uint32_t kFlagcxInt8 = 0;
constexpr uint32_t kFlagcxUint8 = 1;
constexpr uint32_t kFlagcxInt32 = 2;
constexpr uint32_t kFlagcxUint32 = 3;
constexpr uint32_t kFlagcxInt64 = 4;
constexpr uint32_t kFlagcxUint64 = 5;
constexpr uint32_t kFlagcxFloat16 = 6;
constexpr uint32_t kFlagcxFloat32 = 7;
constexpr uint32_t kFlagcxBfloat16 = 9;

constexpr uint32_t kFlagcxSum = 0;
constexpr uint32_t kFlagcxProd = 1;
constexpr uint32_t kFlagcxMax = 2;
constexpr uint32_t kFlagcxMin = 3;
constexpr uint32_t kFlagcxAvg = 4;

constexpr uint64_t kDataCacheLineBytes = 64;

__aicore__ inline bool IsNegativeZero(float value) {
  union {
    float value;
    uint32_t bits;
  } rep;
  rep.value = value;
  return rep.bits == uint32_t(0x80000000);
}

template <typename T>
__aicore__ inline T FloatingMax(T a, T b) {
  if (a != a)
    return b;
  if (b != b)
    return a;
  if (a > b)
    return a;
  if (b > a)
    return b;
  // Match fmax signed-zero behavior without using the SIMT fmax API.
  if (a == static_cast<T>(0) && IsNegativeZero(a))
    return b;
  return a;
}

template <typename T>
__aicore__ inline T FloatingMin(T a, T b) {
  if (a != a)
    return b;
  if (b != b)
    return a;
  if (a < b)
    return a;
  if (b < a)
    return b;
  // Match fmin signed-zero behavior without using the SIMT fmin API.
  if (a == static_cast<T>(0) && IsNegativeZero(b))
    return b;
  return a;
}

template <typename T, typename U, bool Signed>
struct IntegralReducer {
  __aicore__ static inline T Apply(T a, T b, uint32_t redOp,
                                   uint64_t avgDivisor) {
    const U ua = static_cast<U>(a);
    const U ub = static_cast<U>(b);
    switch (redOp) {
      case kFlagcxSum:
        return static_cast<T>(static_cast<U>(ua + ub));
      case kFlagcxProd:
        return static_cast<T>(static_cast<U>(ua * ub));
      case kFlagcxMax:
        return a > b ? a : b;
      case kFlagcxMin:
        return a < b ? a : b;
      case kFlagcxAvg: {
        const T sum = static_cast<T>(static_cast<U>(ua + ub));
        if (Signed) {
          return static_cast<T>(static_cast<int64_t>(sum) /
                                static_cast<int64_t>(avgDivisor));
        } else {
          return static_cast<T>(static_cast<uint64_t>(sum) / avgDivisor);
        }
      }
      default:
        return a;
    }
  }
};

struct FloatReducer {
  __aicore__ static inline float Apply(float a, float b, uint32_t redOp,
                                       uint64_t avgDivisor) {
    switch (redOp) {
      case kFlagcxSum:
        return a + b;
      case kFlagcxProd:
        return a * b;
      case kFlagcxMax:
        return FloatingMax(a, b);
      case kFlagcxMin:
        return FloatingMin(a, b);
      case kFlagcxAvg:
        return (a + b) / static_cast<float>(avgDivisor);
      default:
        return a;
    }
  }
};

struct HalfReducer {
  __aicore__ static inline half Apply(half a, half b, uint32_t redOp,
                                      uint64_t avgDivisor) {
    // Standard Ascend C casts are supported by the A2/A3 scalar path.  Do not
    // use the similarly named SIMT FP16 conversion intrinsics, whose product
    // matrix excludes A2/A3.
    const float af = static_cast<float>(a);
    const float bf = static_cast<float>(b);
    switch (redOp) {
      case kFlagcxSum:
        return static_cast<half>(af + bf);
      case kFlagcxProd:
        return static_cast<half>(af * bf);
      case kFlagcxMax:
        return af > bf ? a : b;
      case kFlagcxMin:
        return af < bf ? a : b;
      case kFlagcxAvg: {
        const half sum = static_cast<half>(af + bf);
        return static_cast<half>(static_cast<float>(sum) /
                                 static_cast<float>(avgDivisor));
      }
      default:
        return a;
    }
  }
};

struct Bfloat16Reducer {
  __aicore__ static inline bfloat16_t Apply(bfloat16_t a, bfloat16_t b,
                                            uint32_t redOp,
                                            uint64_t avgDivisor) {
    const float af = AscendC::ToFloat(a);
    const float bf = AscendC::ToFloat(b);
    switch (redOp) {
      case kFlagcxSum:
        return AscendC::ToBfloat16(af + bf);
      case kFlagcxProd:
        return AscendC::ToBfloat16(af * bf);
      case kFlagcxMax:
        return af > bf ? a : b;
      case kFlagcxMin:
        return af < bf ? a : b;
      case kFlagcxAvg:
        return AscendC::ToBfloat16((af + bf) /
                                   static_cast<float>(avgDivisor));
      default:
        return a;
    }
  }
};

template <typename T>
__aicore__ inline void RefreshCacheLine(AscendC::GlobalTensor<T> tensor) {
  AscendC::DataCacheCleanAndInvalid<
      T, AscendC::CacheLine::SINGLE_CACHE_LINE,
      AscendC::DcciDst::CACHELINE_OUT>(tensor);
}

template <typename T>
__aicore__ inline void RefreshInputLines(AscendC::GlobalTensor<T> tensor,
                                         uint64_t tensorAddress,
                                         uint64_t firstElement,
                                         uint64_t endElement) {
  RefreshCacheLine(tensor[firstElement]);
  const uint64_t firstAddress = tensorAddress + firstElement * sizeof(T);
  const uint64_t lastAddress = tensorAddress + (endElement - 1) * sizeof(T);
  if ((firstAddress / kDataCacheLineBytes) !=
      (lastAddress / kDataCacheLineBytes)) {
    // Refresh from the beginning of the second line.  Refreshing only the
    // final element would leave earlier elements in that line potentially
    // stale because an unaligned SINGLE_CACHE_LINE DCCI covers only the
    // supplied address through the next 64-byte boundary.
    const uint64_t secondLineAddress =
        (firstAddress / kDataCacheLineBytes + uint64_t(1)) *
        kDataCacheLineBytes;
    const uint64_t secondLineElement =
        (secondLineAddress - tensorAddress) / sizeof(T);
    RefreshCacheLine(tensor[secondLineElement]);
  }
}

// GlobalTensor scalar access is deliberately used as a correctness fallback
// for every supported FlagCX datatype/op combination.  To avoid the documented
// multi-core DCache lost-update hazard, each output cache line is owned by
// exactly one core and is explicitly cleaned after it is written.
template <typename T, typename Reducer>
__aicore__ inline void ReduceTyped(GM_ADDR input1, GM_ADDR input2,
                                   GM_ADDR output, uint64_t count,
                                   uint32_t redOp, uint64_t avgDivisor) {
  const uint64_t blockCount = AscendC::GetBlockNum();
  const uint64_t blockIndex = AscendC::GetBlockIdx();
  if (blockCount == 0 || blockIndex >= blockCount)
    return;

  AscendC::GlobalTensor<T> input1Tensor;
  AscendC::GlobalTensor<T> input2Tensor;
  AscendC::GlobalTensor<T> outputTensor;
  input1Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input1),
                               static_cast<uint32_t>(count));
  input2Tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input2),
                               static_cast<uint32_t>(count));
  outputTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output),
                              static_cast<uint32_t>(count));

  const uint64_t input1Address = reinterpret_cast<uint64_t>(input1);
  const uint64_t input2Address = reinterpret_cast<uint64_t>(input2);
  const uint64_t outputAddress = reinterpret_cast<uint64_t>(output);
  const uint64_t outputEnd = outputAddress + count * sizeof(T);
  const uint64_t firstLine =
      outputAddress & ~(kDataCacheLineBytes - uint64_t(1));
  const uint64_t lineEnd =
      (outputEnd + kDataCacheLineBytes - uint64_t(1)) &
      ~(kDataCacheLineBytes - uint64_t(1));
  const uint64_t lineCount = (lineEnd - firstLine) / kDataCacheLineBytes;

  const uint64_t ownedLineBegin =
      firstLine + (lineCount * blockIndex / blockCount) * kDataCacheLineBytes;
  const uint64_t ownedLineEnd =
      firstLine +
      (lineCount * (blockIndex + uint64_t(1)) / blockCount) *
          kDataCacheLineBytes;

  for (uint64_t lineAddress = ownedLineBegin; lineAddress < ownedLineEnd;
       lineAddress += kDataCacheLineBytes) {
    uint64_t elementBegin = 0;
    if (lineAddress > outputAddress)
      elementBegin = (lineAddress - outputAddress) / sizeof(T);

    uint64_t elementEnd = count;
    const uint64_t nextLine = lineAddress + kDataCacheLineBytes;
    if (nextLine < outputEnd)
      elementEnd = (nextLine - outputAddress) / sizeof(T);
    if (elementBegin >= elementEnd)
      continue;

    // Pull the latest contents of a partial first/last output cache line
    // before scalar stores so bytes outside the requested range are
    // preserved when this core writes the line back.
    RefreshCacheLine(outputTensor[elementBegin]);
    RefreshInputLines(input1Tensor, input1Address, elementBegin, elementEnd);
    RefreshInputLines(input2Tensor, input2Address, elementBegin, elementEnd);
    for (uint64_t i = elementBegin; i < elementEnd; ++i) {
      const T a = input1Tensor.GetValue(i);
      const T b = input2Tensor.GetValue(i);
      outputTensor.SetValue(i, Reducer::Apply(a, b, redOp, avgDivisor));
    }
    RefreshCacheLine(outputTensor[elementBegin]);
  }
}

} // namespace

extern "C" __global__ __aicore__ void flagcx_ascend_unirunner_reduce(
    GM_ADDR input1, GM_ADDR input2, GM_ADDR output, uint64_t count,
    uint32_t datatype, uint32_t redOp, uint64_t avgDivisor) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  switch (datatype) {
    case kFlagcxInt8:
      ReduceTyped<int8_t, IntegralReducer<int8_t, uint8_t, true>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxUint8:
      ReduceTyped<uint8_t, IntegralReducer<uint8_t, uint8_t, false>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxInt32:
      ReduceTyped<int32_t, IntegralReducer<int32_t, uint32_t, true>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxUint32:
      ReduceTyped<uint32_t, IntegralReducer<uint32_t, uint32_t, false>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxInt64:
      ReduceTyped<int64_t, IntegralReducer<int64_t, uint64_t, true>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxUint64:
      ReduceTyped<uint64_t, IntegralReducer<uint64_t, uint64_t, false>>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    case kFlagcxFloat16:
      ReduceTyped<half, HalfReducer>(input1, input2, output, count, redOp,
                                     avgDivisor);
      break;
    case kFlagcxFloat32:
      ReduceTyped<float, FloatReducer>(input1, input2, output, count, redOp,
                                       avgDivisor);
      break;
    case kFlagcxBfloat16:
      ReduceTyped<bfloat16_t, Bfloat16Reducer>(
          input1, input2, output, count, redOp, avgDivisor);
      break;
    default:
      break;
  }
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void flagcx_ascend_unirunner_reduce_do(
    uint32_t blockDim, void *stream, uint8_t *input1, uint8_t *input2,
    uint8_t *output, uint64_t count, uint32_t datatype, uint32_t redOp,
    uint64_t avgDivisor) {
  flagcx_ascend_unirunner_reduce<<<blockDim, nullptr, stream>>>(
      input1, input2, output, count, datatype, redOp, avgDivisor);
}
#endif
