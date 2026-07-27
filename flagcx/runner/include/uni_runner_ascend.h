/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_UNI_RUNNER_ASCEND_H_
#define FLAGCX_UNI_RUNNER_ASCEND_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Launch one element-wise binary reduction on an Ascend stream.
//
// This is the short-lived-kernel fallback used by the Ascend UniRunner
// backend.  It intentionally does not use the CUDA-oriented persistent FIFO
// executor because Atlas A2/A3 do not expose the general system-scope
// CAS/fetch-add primitives required by that executor.
// FP64 reduction is rejected with flagcxNotSupported: the target A2/A3
// device compiler does not provide a portable FP64 scalar-arithmetic path.
flagcxResult_t flagcxAscendUniRunnerLaunchReduce(
    const void *input1, const void *input2, void *output, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, uint64_t avgDivisor,
    size_t nBlocks, flagcxStream_t stream);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FLAGCX_UNI_RUNNER_ASCEND_H_
