/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "flagcx_hetero.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_ascend.h"
#include "uni_runner_impl.h"

FLAGCX_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
FLAGCX_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);
FLAGCX_PARAM(UniRunnerUseSlicedAR, "UNIRUNNER_USE_SLICEDAR", 0);
FLAGCX_PARAM(UniRunnerUseIpcAR, "UNIRUNNER_USE_IPCAR", 0);
FLAGCX_PARAM(UniRunnerUseGroupedAG, "UNIRUNNER_USE_GROUPEDAG", 1);
FLAGCX_PARAM(UniRunnerGroupSize, "UNIRUNNER_GROUPSIZE", 0);
#ifdef USE_ASCEND_ADAPTOR
FLAGCX_PARAM(UniRunnerUseHccsA2A, "UNIRUNNER_USE_HCCSA2A", 0);
FLAGCX_PARAM(UniRunnerUseIpcA2A, "UNIRUNNER_USE_IPCA2A", 0);
#endif

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
#ifdef USE_ASCEND_ADAPTOR
  bool runnerInitialized = false;
  bool scratchSafeToFree = false;
#endif
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
#ifdef USE_ASCEND_ADAPTOR
  if (count == 0)
    return flagcxSuccess;
#endif
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 2, datatype, &scratchBytes));
#ifdef USE_ASCEND_ADAPTOR
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    return res;
  runnerInitialized = true;
  res = deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                    flagcxMemDevice, stream);
  if (res != flagcxSuccess)
    goto out;
#else
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                          flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
#endif
  FLAGCXCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
#ifdef USE_ASCEND_ADAPTOR
  if (runnerInitialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (cleanupRes != flagcxSuccess) {
      res = cleanupRes;
    } else {
      scratchSafeToFree = true;
    }
  }
  if (scratchbuff != nullptr && scratchSafeToFree) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream);
    if (freeRes != flagcxSuccess)
      res = freeRes;
  } else if (scratchbuff != nullptr) {
    WARN("Ascend UniRunner: retaining Reduce scratch buffer %p after "
         "stream/cleanup failure",
         scratchbuff);
  }
#else
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
#endif
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
  const void *runnerSendbuff = sendbuff;
#ifdef USE_ASCEND_ADAPTOR
  void *stagedSendbuff = nullptr;
  size_t stagedBytes = 0;
  bool runnerInitialized = false;
  bool stagedSafeToFree = false;
#endif
  FLAGCXCHECK(validateUniRunnerReduceArgs(count, datatype, op));
#ifdef USE_ASCEND_ADAPTOR
  if (count == 0)
    return flagcxSuccess;
  if (sendbuff == recvbuff && count != 0) {
    FLAGCXCHECK(
        checkedUniRunnerTypeBytes(count, 1, datatype, &stagedBytes));
  }
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    goto out;
  runnerInitialized = true;

  // The existing SlicedAR topology receives peer chunks into recvbuff before
  // their local reduction reads sendbuff. Preserve the public in-place
  // contract by snapshotting the original input; this also makes IPCAR's
  // capability fallback to SlicedAR safe without changing either DAG.
  if (sendbuff == recvbuff && count != 0) {
    res = deviceAdaptor->deviceMalloc(&stagedSendbuff, stagedBytes,
                                      flagcxMemDevice, stream);
    if (res != flagcxSuccess)
      goto out;
    res = deviceAdaptor->deviceMemcpy(
        stagedSendbuff, const_cast<void *>(sendbuff), stagedBytes,
        flagcxMemcpyDeviceToDevice, stream, NULL);
    if (res != flagcxSuccess)
      goto out;
    runnerSendbuff = stagedSendbuff;
  }
#else
  FLAGCXCHECK(initUniRunner(comm, stream));
#endif
  if (flagcxParamUniRunnerUseIpcAR()) {
    /* Sliced AllReduce with intra-node IPC/LSA push transport. */
    FLAGCXCHECKGOTO(initUniRunnerStateIpcAR(runnerState, runnerSendbuff,
                                            recvbuff, count, datatype, op,
                                            comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseLocRed()) {
    /* initialize uniRunnerState for reduce test */
    FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, runnerSendbuff,
                                             recvbuff, count, datatype, op,
                                             comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseRingAG()) {
    /* initialize uniRunnerState for p2p test */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, runnerSendbuff,
                                             recvbuff, count, datatype, op,
                                             comm),
                    res, out);
  } else if (flagcxParamUniRunnerUseSlicedAR()) {
    /* initialize uniRunnerState for sliced AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(
                        runnerState, runnerSendbuff, recvbuff, count, datatype,
                        op, comm),
                    res, out);
  } else {
    /* initialize uniRunnerState for ring AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, runnerSendbuff,
                                             recvbuff, count, datatype, op,
                                             comm),
                    res, out);
  }
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
#ifdef USE_ASCEND_ADAPTOR
  if (runnerInitialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (cleanupRes != flagcxSuccess) {
      res = cleanupRes;
    } else {
      stagedSafeToFree = true;
    }
  } else if (stagedSendbuff != nullptr) {
    // The staging copy is asynchronous on the user stream. If runner
    // initialization failed before normal cleanup could synchronize it, wait
    // before releasing the temporary allocation.
    flagcxResult_t syncRes = deviceAdaptor->streamSynchronize(stream);
    if (syncRes != flagcxSuccess) {
      res = syncRes;
    } else {
      stagedSafeToFree = true;
    }
  }
  if (stagedSendbuff != nullptr && stagedSafeToFree) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(stagedSendbuff, flagcxMemDevice, stream);
    if (freeRes != flagcxSuccess)
      res = freeRes;
  } else if (stagedSendbuff != nullptr) {
    WARN("Ascend UniRunner: retaining in-place staging buffer %p after "
         "stream/cleanup failure",
         stagedSendbuff);
  }
#else
  FLAGCXCHECK(cleanupUniRunner(comm));
#endif
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
#ifdef USE_ASCEND_ADAPTOR
  bool runnerInitialized = false;
  bool scratchSafeToFree = false;
#endif
  FLAGCXCHECK(validateUniRunnerReduceArgs(recvcount, datatype, op));
#ifdef USE_ASCEND_ADAPTOR
  if (recvcount == 0)
    return flagcxSuccess;
#endif
  FLAGCXCHECK(checkedUniRunnerTypeBytes(recvcount, comm->nranks, datatype,
                                        &scratchBytes));
#ifdef USE_ASCEND_ADAPTOR
  res = initUniRunner(comm, stream);
  if (res != flagcxSuccess)
    return res;
  runnerInitialized = true;
  res = deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                    flagcxMemDevice, stream);
  if (res != flagcxSuccess)
    goto out;
#else
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&scratchbuff, scratchBytes,
                                          flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(comm, stream), res, out);
#endif
  FLAGCXCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           comm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(comm), res, out);
out:
#ifdef USE_ASCEND_ADAPTOR
  if (runnerInitialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (cleanupRes != flagcxSuccess) {
      res = cleanupRes;
    } else {
      scratchSafeToFree = true;
    }
  }
  if (scratchbuff != nullptr && scratchSafeToFree) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream);
    if (freeRes != flagcxSuccess)
      res = freeRes;
  } else if (scratchbuff != nullptr) {
    WARN("Ascend UniRunner: retaining ReduceScatter scratch buffer %p after "
         "stream/cleanup failure",
         scratchbuff);
  }
#else
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(comm));
#endif
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

#ifdef USE_ASCEND_ADAPTOR
static flagcxResult_t
ascendUniRunnerIpcPeerAlltoAll(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxComm_t comm, flagcxStream_t stream) {
  if (comm == NULL || stream == NULL || comm->nranks < 1 || comm->rank < 0 ||
      comm->rank >= comm->nranks) {
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
  if (flagcxGroupDepth != 0) {
    WARN("Ascend UniRunner ACL IPC peer-pointer AlltoAll does not support "
         "FlagCX group capture; refusing to fall back to another transport");
    return flagcxNotSupported;
  }

  if (comm->nranks == 1) {
    if (sendbuff == recvbuff) {
      return flagcxSuccess;
    }
    return deviceAdaptor->deviceMemcpy(
        recvbuff, const_cast<void *>(sendbuff), totalBytes,
        flagcxMemcpyDeviceToDevice, stream, NULL);
  }
  if (comm->heteroComm == NULL || comm->heteroComm->proxyState == NULL) {
    WARN("Ascend UniRunner ACL IPC peer-pointer AlltoAll requires "
         "FLAGCX_USE_HETERO_COMM=1");
    return flagcxInvalidUsage;
  }

  flagcxResult_t res = flagcxSuccess;
  flagcxUniRunnerState *runnerState =
      &comm->heteroComm->proxyState->uniRunnerState;
  const void *effectiveSendbuff = sendbuff;
  void *scratchbuff = NULL;
  bool runnerInitialized = false;
  bool scratchSafeToFree = false;
  flagcxResult_t collectiveRes = flagcxSuccess;

  // AlltoAll permits sendbuff == recvbuff. Snapshot the complete input before
  // any peer writes into this rank's imported receive-buffer pointer.
  if (sendbuff == recvbuff) {
    res = deviceAdaptor->deviceMalloc(&scratchbuff, totalBytes,
                                      flagcxMemDevice, stream);
    if (res == flagcxSuccess) {
      res = deviceAdaptor->deviceMemcpy(
          scratchbuff, const_cast<void *>(sendbuff), totalBytes,
          flagcxMemcpyDeviceToDevice, stream, NULL);
      if (res == flagcxSuccess)
        effectiveSendbuff = scratchbuff;
    }
  }

  // Snapshot allocation/copy can fail independently on one device. Agree
  // before any rank initializes the collective IPC state.
  {
    flagcxResult_t agreeRes =
        agreeAscendUniRunnerIpcResult(comm, res, &collectiveRes);
    if (agreeRes != flagcxSuccess) {
      res = agreeRes;
      goto out;
    }
    if (collectiveRes != flagcxSuccess) {
      res = collectiveRes;
      goto out;
    }
  }

  res = initUniRunner(comm, stream);
  runnerInitialized = res == flagcxSuccess;
  {
    // Stream creation is also rank-local. A successful rank must clean up its
    // streams rather than entering IPC setup when a peer failed initialization.
    flagcxResult_t agreeRes =
        agreeAscendUniRunnerIpcResult(comm, res, &collectiveRes);
    if (agreeRes != flagcxSuccess) {
      res = agreeRes;
      goto out;
    }
    if (collectiveRes != flagcxSuccess) {
      res = collectiveRes;
      goto out;
    }
  }
  if (!runnerInitialized) {
    res = flagcxInternalError;
    goto out;
  }

  res = initUniRunnerStateIpcA2A(runnerState, effectiveSendbuff, recvbuff,
                                 count, datatype, comm);
  if (res != flagcxSuccess)
    goto out;
  INFO(FLAGCX_UNIRUNNER,
       "Ascend UniRunner AlltoAll "
       "selected=ASCEND_IPC_A2A_PEER_COPY runner=UNIRUNNER rank=%d "
       "peer_bytes=%zu transport=aclrtMemcpyAsync acl_ipc=1 peer_pointer=1 "
       "pid_whitelist=1 peer_access=explicit ready_signal=peer_d2d "
       "output_export=1 input_export=0 fail_if_unavailable=1 "
       "hccl_collective=0 hcomm=0 socket=0 host_staging=0",
       comm->rank, totalBytes / static_cast<size_t>(comm->nranks));
  res = runUniRunner(comm);

out:
  if (runnerInitialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (cleanupRes == flagcxSuccess) {
      scratchSafeToFree = true;
    } else if (res == flagcxSuccess) {
      res = cleanupRes;
    }
  } else if (scratchbuff != NULL) {
    // The in-place snapshot may already be queued on the caller stream.
    flagcxResult_t syncRes = deviceAdaptor->streamSynchronize(stream);
    if (syncRes == flagcxSuccess) {
      scratchSafeToFree = true;
    } else if (res == flagcxSuccess) {
      res = syncRes;
    }
  }

  if (scratchbuff != NULL && scratchSafeToFree) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream);
    if (res == flagcxSuccess)
      res = freeRes;
  } else if (scratchbuff != NULL) {
    WARN("Ascend UniRunner ACL IPC AlltoAll: retaining in-place snapshot %p "
         "after stream or IPC cleanup failure",
         scratchbuff);
  }
  return res;
}
#endif

flagcxResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxComm_t comm, flagcxStream_t stream) {
#ifdef USE_ASCEND_ADAPTOR
  const bool useHccsA2A = flagcxParamUniRunnerUseHccsA2A() != 0;
  const bool useIpcA2A = flagcxParamUniRunnerUseIpcA2A() != 0;
  if (useHccsA2A && useIpcA2A) {
    WARN("Ascend UniRunner AlltoAll transport selection is ambiguous: "
         "UNIRUNNER_USE_HCCSA2A and UNIRUNNER_USE_IPCA2A cannot both be set");
    return flagcxInvalidUsage;
  }
  if (useHccsA2A) {
    if (flagcxGroupDepth != 0) {
      WARN("Ascend UniRunner HCCS AlltoAll does not support FlagCX group "
           "capture; refusing to fall back to another transport");
      return flagcxNotSupported;
    }
    return flagcxAscendUniRunnerHccsAlltoAll(sendbuff, recvbuff, count,
                                             datatype, comm, stream);
  }
  if (useIpcA2A) {
    return ascendUniRunnerIpcPeerAlltoAll(sendbuff, recvbuff, count, datatype,
                                          comm, stream);
  }
#endif
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(bufferIn + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
    FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
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
