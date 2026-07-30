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

// Prepare communicator-scoped HCOMM/HCCS resources independently from any
// collective algorithm. P2P DAG nodes subsequently submit send/recv
// primitives through this group API.
flagcxResult_t flagcxAscendUniRunnerHccsPrepare(flagcxComm_t comm);
flagcxResult_t flagcxAscendUniRunnerHccsGroupStart(flagcxComm_t comm,
                                                  flagcxStream_t stream);
flagcxResult_t flagcxAscendUniRunnerHccsSend(const void *sendbuff,
                                            size_t count,
                                            flagcxDataType_t datatype,
                                            int peer, flagcxComm_t comm);
flagcxResult_t flagcxAscendUniRunnerHccsRecv(void *recvbuff, size_t count,
                                            flagcxDataType_t datatype,
                                            int peer, flagcxComm_t comm);
flagcxResult_t flagcxAscendUniRunnerHccsGroupEnd(flagcxComm_t comm);

// Internal symmetric-group executor retained behind the transport interface.
// Collective entry points must use the P2P DAG and group API above.
flagcxResult_t flagcxAscendUniRunnerHccsAlltoAll(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxComm_t comm, flagcxStream_t stream);

// HCCL owns ThreadHandle/ChannelHandle until communicator destruction, while
// FlagCX owns the ACL stream bound to the thread. Teardown is therefore
// split around HcclCommDestroy.
flagcxResult_t
flagcxAscendUniRunnerHccsPrepareDestroy(flagcxComm_t comm);
flagcxResult_t
flagcxAscendUniRunnerHccsFinishDestroy(flagcxComm_t comm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FLAGCX_UNI_RUNNER_ASCEND_H_
