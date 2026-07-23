/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "flagcx_hetero.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_impl.h"

FLAGCX_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
FLAGCX_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);
FLAGCX_PARAM(UniRunnerUseSlicedAR, "UNIRUNNER_USE_SLICEDAR", 0);
FLAGCX_PARAM(UniRunnerUseIpcAR, "UNIRUNNER_USE_IPCAR", 0);
FLAGCX_PARAM(UniRunnerUseGroupedAG, "UNIRUNNER_USE_GROUPEDAG", 1);
FLAGCX_PARAM(UniRunnerGroupSize, "UNIRUNNER_GROUPSIZE", 0);

static int resolveUniRunnerGroupedAGGroupSize(flagcxComm_t comm) {
  if (comm->nranks <= 0) {
    return 0;
  }

  int groupSize = flagcxParamUniRunnerGroupSize();
  if (groupSize <= 0) {
    groupSize = comm->localRanks > 1 ? comm->localRanks : comm->nranks;
  }
  if (groupSize <= 0 || groupSize > comm->nranks ||
      comm->nranks % groupSize != 0) {
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d groupedAG groupSize %d invalid for nranks %d, fallback to "
          "nranks",
          comm->rank, groupSize, comm->nranks);
    groupSize = comm->nranks;
  }
  return groupSize;
}

flagcxResult_t uniRunnerReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  size_t scratchBytes = 0;
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 2, datatype, &scratchBytes));
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                          flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

flagcxResult_t uniRunnerGather(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerScatter(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, r,
                                   comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
  FLAGCXCHECK(initUniRunner(comm, stream));
  if (flagcxParamUniRunnerUseIpcAR()) {
    /* Sliced AllReduce with intra-node IPC/LSA push transport. */
    FLAGCXCHECKGOTO(initUniRunnerStateIpcAR(runnerState, sendbuff, recvbuff,
                                            count, datatype, op, comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseLocRed()) {
    /* initialize uniRunnerState for reduce test */
    FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseRingAG()) {
    /* initialize uniRunnerState for p2p test */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseSlicedAR()) {
    /* initialize uniRunnerState for sliced AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, comm),
                    res, out);
  } else {
    /* initialize uniRunnerState for ring AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, comm),
                    res, out);
  }
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

flagcxResult_t uniRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxComm_t comm,
                                      flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  size_t scratchBytes = 0;
  FLAGCXCHECK(validateUniRunnerReduceArgs(recvcount, datatype, op));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(recvcount, comm->nranks, datatype,
                                        &scratchBytes));
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                          flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  if (!flagcxParamUniRunnerUseGroupedAG()) {
    size_t size = sendcount * getFlagcxDataTypeSize(datatype);
    char *bufferOut = static_cast<char *>(recvbuff);
    FLAGCXCHECK(flagcxHeteroGroupStart());
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(sendbuff, sendcount, datatype, r,
                                   comm->heteroComm, stream));
      FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                   sendcount, datatype, r, comm->heteroComm,
                                   stream));
    }
    FLAGCXCHECK(flagcxHeteroGroupEnd());
    return flagcxSuccess;
  }

  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  int groupSize = resolveUniRunnerGroupedAGGroupSize(comm);

  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff,
                                              sendcount, datatype, comm,
                                              groupSize),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
  FLAGCXCHECK(cleanupUniRunner(comm));
  return res;
}

static flagcxResult_t enqueueGroupedUniRunnerAlltoAll(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, size_t blockBytes, flagcxComm_t comm,
    flagcxStream_t stream, int opId = INT_MAX, int step = -1,
    bool includeSelf = true) {
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int peer = 0; peer < comm->nranks; ++peer) {
    if (!includeSelf && peer == comm->rank) {
      continue;
    }
    FLAGCXCHECK(flagcxHeteroSend(
        static_cast<const void *>(bufferIn +
                                  static_cast<size_t>(peer) * blockBytes),
        count, datatype, peer, comm->heteroComm, stream, opId, step));
    FLAGCXCHECK(flagcxHeteroRecv(
        static_cast<void *>(bufferOut +
                            static_cast<size_t>(peer) * blockBytes),
        count, datatype, peer, comm->heteroComm, stream, opId, step));
  }
  return flagcxHeteroGroupEnd();
}

flagcxResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxComm_t comm, flagcxStream_t stream) {
  if (comm == NULL || comm->nranks < 1) {
    return flagcxInvalidArgument;
  }

  size_t totalBytes = 0;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      count, static_cast<size_t>(comm->nranks), datatype, &totalBytes));
  if (totalBytes == 0) {
    return flagcxSuccess;
  }
  if (sendbuff == NULL || recvbuff == NULL) {
    return flagcxInvalidArgument;
  }
  if (comm->nranks == 1 && sendbuff == recvbuff) {
    return flagcxSuccess;
  }

  // An outer FlagCX group owns submission and buffer lifetime. Queue the
  // operation directly so the outer GroupEnd launches it; runUniRunner cannot
  // drain a nested group.
  if (flagcxGroupDepth > 0) {
    const size_t blockBytes =
        totalBytes / static_cast<size_t>(comm->nranks);
    if (sendbuff == recvbuff) {
      // Preserve the group contract that no stream work starts before
      // GroupEnd. The self P2P pair snapshots the whole input at step 0; all
      // pairwise exchanges wait at step 1. GroupEnd defers freeing the scratch
      // allocation until its completion callback/kernel has drained.
      void *groupScratch = NULL;
      FLAGCXCHECK(deviceAdaptor->deviceMalloc(
          &groupScratch, totalBytes, flagcxMemDevice, NULL));
      flagcxResult_t deferResult =
          flagcxGroupDeferFree(groupScratch, flagcxMemDevice, stream);
      if (deferResult != flagcxSuccess) {
        (void)deviceAdaptor->deviceFree(groupScratch, flagcxMemDevice, NULL);
        return deferResult;
      }

      const int opId = flagcxGroupAllocCustomOpId();
      const size_t typeSize = getFlagcxDataTypeSize(datatype);
      const size_t totalCount = totalBytes / typeSize;
      FLAGCXCHECK(flagcxHeteroSend(sendbuff, totalCount, datatype, comm->rank,
                                   comm->heteroComm, stream, opId, 0));
      FLAGCXCHECK(flagcxHeteroRecv(groupScratch, totalCount, datatype,
                                   comm->rank, comm->heteroComm, stream, opId,
                                   0));
      return enqueueGroupedUniRunnerAlltoAll(
          groupScratch, recvbuff, count, datatype, blockBytes, comm, stream,
          opId, 1, false);
    }
    return enqueueGroupedUniRunnerAlltoAll(sendbuff, recvbuff, count, datatype,
                                           blockBytes, comm, stream);
  }

  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  const void *effectiveSendbuff = sendbuff;
  void *scratchbuff = NULL;
  bool runnerInitialized = false;

  // AlltoAll permits sendbuff == recvbuff. Snapshot the complete input before
  // launching direct pairwise receives so incoming blocks cannot overwrite
  // blocks that have not yet been sent.
  if (sendbuff == recvbuff) {
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMalloc(
                        &scratchbuff, totalBytes, flagcxMemDevice, stream),
                    res, out);
    FLAGCXCHECKGOTO(deviceAdaptor->deviceMemcpy(
                        scratchbuff, const_cast<void *>(sendbuff), totalBytes,
                        flagcxMemcpyDeviceToDevice, stream, NULL),
                    res, out);
    FLAGCXCHECKGOTO(deviceAdaptor->streamSynchronize(stream), res, out);
    effectiveSendbuff = scratchbuff;
  }

  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
  runnerInitialized = true;
  FLAGCXCHECKGOTO(initUniRunnerStateAlltoAll(
                      runnerState, effectiveSendbuff, recvbuff, count, datatype,
                      comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);

out:
  if (runnerInitialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (res == flagcxSuccess) {
      res = cleanupRes;
    }
  }
  if (scratchbuff != NULL) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream);
    if (res == flagcxSuccess) {
      res = freeRes;
    }
  }
  return res;
}

flagcxResult_t uniRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECK(flagcxHeteroSend(
          static_cast<const void *>(bufferIn + sdispls[r] * size),
          sendcounts[r], datatype, r, comm->heteroComm, stream));
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECK(flagcxHeteroRecv(
          static_cast<void *>(bufferOut + rdispls[r] * size), recvcounts[r],
          datatype, r, comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerSend(const void *sendbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerRecv(void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupStart() {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupEnd() {
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

struct flagcxRunner uniRunner = {
    // Communication functions
    uniRunnerReduce, uniRunnerGather, uniRunnerScatter, uniRunnerBroadcast,
    uniRunnerAllReduce, uniRunnerReduceScatter, uniRunnerAllGather,
    uniRunnerAlltoAll, uniRunnerAlltoAllv, uniRunnerSend, uniRunnerRecv,
    // Group semantics
    uniRunnerGroupStart, uniRunnerGroupEnd};
