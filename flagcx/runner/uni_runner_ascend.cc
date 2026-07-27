/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "uni_runner_ascend.h"

#include <cstdint>
#include <limits>

#if defined(USE_ASCEND_ADAPTOR)
#include "ascend_adaptor.h"
#endif

namespace {

static size_t ascendReduceTypeSize(flagcxDataType_t datatype) {
  switch (datatype) {
    case flagcxInt8:
    case flagcxUint8:
      return 1;
    case flagcxFloat16:
    case flagcxBfloat16:
      return 2;
    case flagcxInt32:
    case flagcxUint32:
    case flagcxFloat32:
      return 4;
    case flagcxInt64:
    case flagcxUint64:
    case flagcxFloat64:
      return 8;
    default:
      return 0;
  }
}

static bool ascendReduceOpSupported(flagcxRedOp_t redOp) {
  return redOp == flagcxSum || redOp == flagcxProd || redOp == flagcxMax ||
         redOp == flagcxMin || redOp == flagcxAvg;
}

} // namespace

#if defined(USE_ASCEND_ADAPTOR) && defined(COMPILE_KERNEL_HOST)

// Implemented by the host half of the ascendc_library target in
// flagcx/kernels/ascend.  The generated launcher submits asynchronously to
// the supplied ACL stream.
extern "C" void flagcx_ascend_unirunner_reduce_do(
    uint32_t blockDim, void *stream, uint8_t *input1, uint8_t *input2,
    uint8_t *output, uint64_t count, uint32_t datatype, uint32_t redOp,
    uint64_t avgDivisor);

#endif

extern "C" flagcxResult_t flagcxAscendUniRunnerLaunchReduce(
    const void *input1, const void *input2, void *output, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, uint64_t avgDivisor,
    size_t nBlocks, flagcxStream_t stream) {
#if !defined(USE_ASCEND_ADAPTOR) || !defined(COMPILE_KERNEL_HOST)
  (void)input1;
  (void)input2;
  (void)output;
  (void)count;
  (void)datatype;
  (void)redOp;
  (void)avgDivisor;
  (void)nBlocks;
  (void)stream;
  return flagcxNotSupported;
#else
  constexpr size_t kMaxAscendBlockDim = 65535;

  // Keep the unsupported capability distinguishable from malformed input.
  // A2/A3 CANN device compilation does not expose a portable FP64 scalar
  // arithmetic path for this correctness fallback.
  if (datatype == flagcxFloat64)
    return flagcxNotSupported;

  // Match the zero-count behavior of the other device backends without
  // requiring otherwise-unused buffer or stream handles.
  if (count == 0)
    return flagcxSuccess;

  const size_t typeSize = ascendReduceTypeSize(datatype);
  if (input1 == nullptr || input2 == nullptr || output == nullptr ||
      stream == nullptr || typeSize == 0 || !ascendReduceOpSupported(redOp) ||
      nBlocks == 0 ||
      nBlocks > kMaxAscendBlockDim ||
      count > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      count > std::numeric_limits<size_t>::max() / typeSize ||
      (redOp == flagcxAvg && avgDivisor == 0)) {
    return flagcxInvalidArgument;
  }

  const size_t bytes = count * typeSize;
  constexpr uintptr_t kCacheLinePadding = 63;
  if (bytes > std::numeric_limits<uintptr_t>::max() - kCacheLinePadding ||
      reinterpret_cast<uintptr_t>(input1) >
          std::numeric_limits<uintptr_t>::max() - bytes ||
      reinterpret_cast<uintptr_t>(input2) >
          std::numeric_limits<uintptr_t>::max() - bytes ||
      reinterpret_cast<uintptr_t>(output) >
          std::numeric_limits<uintptr_t>::max() - bytes -
              kCacheLinePadding) {
    return flagcxInvalidArgument;
  }

  // Scalar GM accesses in the correctness fallback are naturally aligned.
  // Reject malformed typed pointers rather than relying on device-side
  // unaligned scalar behavior.
  if ((reinterpret_cast<uintptr_t>(input1) % typeSize) != 0 ||
      (reinterpret_cast<uintptr_t>(input2) % typeSize) != 0 ||
      (reinterpret_cast<uintptr_t>(output) % typeSize) != 0) {
    return flagcxInvalidArgument;
  }

  static_assert(flagcxInt8 == 0 && flagcxUint8 == 1 && flagcxInt32 == 2 &&
                    flagcxUint32 == 3 && flagcxInt64 == 4 &&
                    flagcxUint64 == 5 && flagcxFloat16 == 6 &&
                    flagcxFloat32 == 7 && flagcxFloat64 == 8 &&
                    flagcxBfloat16 == 9,
                "Ascend UniRunner kernel datatype ABI changed");
  static_assert(flagcxSum == 0 && flagcxProd == 1 && flagcxMax == 2 &&
                    flagcxMin == 3 && flagcxAvg == 4,
                "Ascend UniRunner kernel reduction-op ABI changed");

  // Runtime last-error state is thread-local and sticky.  A recoverable
  // earlier ACL probe (for example IPC capability fallback) must not be
  // mistaken for this launch's result.
  (void)aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  flagcx_ascend_unirunner_reduce_do(
      static_cast<uint32_t>(nBlocks), static_cast<void *>(stream->base),
      const_cast<uint8_t *>(static_cast<const uint8_t *>(input1)),
      const_cast<uint8_t *>(static_cast<const uint8_t *>(input2)),
      static_cast<uint8_t *>(output), static_cast<uint64_t>(count),
      static_cast<uint32_t>(datatype), static_cast<uint32_t>(redOp),
      avgDivisor);
  // The Ascend C triple-chevron launcher is generated with a void return
  // type.  Recover its thread-local Runtime submission status before the
  // caller enqueues the dependent DONE flag write.
  const aclError launchResult = aclrtGetLastError(ACL_RT_THREAD_LEVEL);
  return launchResult == ACL_SUCCESS ? flagcxSuccess
                                     : flagcxUnhandledDeviceError;
#endif
}
