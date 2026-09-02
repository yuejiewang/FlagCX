#include "uni_runner_impl.h"
#include "adaptor.h"
#include "comm.h"
#include "global_comm.h"
#include "proxy.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

FLAGCX_PARAM(UniRunnerFusedAgGemmAlgo, "UNIRUNNER_FUSED_AG_GEMM_ALGO", 0);
FLAGCX_PARAM(UniRunnerFusedGemmRsAlgo, "UNIRUNNER_FUSED_GEMM_RS_ALGO", 0);
FLAGCX_PARAM(UniRunnerGemmKSlices, "UNIRUNNER_GEMM_KSLICES", 1);

namespace {

static bool checkedMul(size_t a, size_t b, size_t *result) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    return false;
  }
  *result = a * b;
  return true;
}

static bool checkedAdd(size_t a, size_t b, size_t *result) {
  if (b > std::numeric_limits<size_t>::max() - a) {
    return false;
  }
  *result = a + b;
  return true;
}

static bool rangesOverlap(const void *first, size_t firstBytes,
                          const void *second, size_t secondBytes,
                          bool *valid) {
  uintptr_t firstBegin = reinterpret_cast<uintptr_t>(first);
  uintptr_t secondBegin = reinterpret_cast<uintptr_t>(second);
  if (firstBytes > std::numeric_limits<uintptr_t>::max() - firstBegin ||
      secondBytes > std::numeric_limits<uintptr_t>::max() - secondBegin) {
    *valid = false;
    return false;
  }
  *valid = true;
  uintptr_t firstEnd = firstBegin + firstBytes;
  uintptr_t secondEnd = secondBegin + secondBytes;
  return firstBegin < secondEnd && secondBegin < firstEnd;
}

static flagcxResult_t validateBufferRanges(const void *input,
                                           size_t inputBytes,
                                           const void *weight,
                                           size_t weightBytes, void *output,
                                           size_t outputBytes) {
  bool valid = false;
  if (rangesOverlap(input, inputBytes, weight, weightBytes, &valid) || !valid ||
      rangesOverlap(input, inputBytes, output, outputBytes, &valid) || !valid ||
      rangesOverlap(weight, weightBytes, output, outputBytes, &valid) ||
      !valid) {
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static flagcxResult_t getEffectiveSlices(size_t k, size_t *slices) {
  int64_t requested = flagcxParamUniRunnerGemmKSlices();
  if (requested <= 0) {
    return flagcxInvalidArgument;
  }
  *slices = std::min(k, static_cast<size_t>(requested));
  return flagcxSuccess;
}

static size_t kSliceOffset(size_t k, size_t slices, size_t slice) {
  size_t base = k / slices;
  size_t remainder = k % slices;
  return slice * base + std::min(slice, remainder);
}

static size_t kSliceLength(size_t k, size_t slices, size_t slice) {
  size_t base = k / slices;
  return base + (slice < k % slices ? 1 : 0);
}

static flagcxResult_t initializeDag(flagcxUniRunnerState *runnerState,
                                    size_t numNodes) {
  if (numNodes > static_cast<size_t>(INT_MAX)) {
    return flagcxInvalidArgument;
  }
  size_t dagBytes = 0;
  if (!checkedMul(numNodes, sizeof(uniRunnerDagNode), &dagBytes)) {
    return flagcxInvalidArgument;
  }
  runnerState->numDagNodes = static_cast<int>(numNodes);
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, dagBytes));
  return runnerState->dagNodes == NULL ? flagcxSystemError : flagcxSuccess;
}

static void enqueueInitialNodes(flagcxUniRunnerState *runnerState) {
  runnerState->numPendingNodes = 0;
  for (int i = 0; i < runnerState->numDagNodes; ++i) {
    uniRunnerDagNode *node = &runnerState->dagNodes[i];
    if (node->numParents == 0) {
      if (node->nodeType == uniRunnerDagNodeTypeP2p ||
          node->nodeType == uniRunnerDagNodeTypeCpy) {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, node);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }
}

static void setGemmNode(uniRunnerDagNode *node, int nodeIdx, const void *a,
                        const void *b, void *c, uint32_t m, uint32_t n,
                        uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc,
                        size_t nthreads, int accumulate, int numParents,
                        int numChildren) {
  node->nodeIdx = nodeIdx;
  node->nodeType = uniRunnerDagNodeTypeGemm;
  node->numParents = numParents;
  node->numChildren = numChildren;
  node->nodeData.gemm.a = a;
  node->nodeData.gemm.b = b;
  node->nodeData.gemm.c = c;
  node->nodeData.gemm.m = m;
  node->nodeData.gemm.n = n;
  node->nodeData.gemm.k = k;
  node->nodeData.gemm.lda = lda;
  node->nodeData.gemm.ldb = ldb;
  node->nodeData.gemm.ldc = ldc;
  node->nodeData.gemm.nthreads = nthreads;
  node->nodeData.gemm.datatype = flagcxFloat32;
  node->nodeData.gemm.accumulate = accumulate;
  node->nodeData.gemm.triggerIdx = -1;
}

static flagcxResult_t setP2pNode(uniRunnerDagNode *node, int nodeIdx,
                                 int numParents, int numChildren, void *send,
                                 void *recv, size_t count, int next, int prev) {
  node->nodeIdx = nodeIdx;
  node->nodeType = uniRunnerDagNodeTypeP2p;
  node->numParents = numParents;
  node->numChildren = numChildren;
  node->nodeData.p2p.numOps = 2;
  FLAGCXCHECK(flagcxCalloc(&node->nodeData.p2p.ops,
                           2 * sizeof(uniRunnerP2pOpData)));
  node->nodeData.p2p.ops[0] =
      {send, count, next, flagcxFloat32, flagcxDevicePrimSend};
  node->nodeData.p2p.ops[1] =
      {recv, count, prev, flagcxFloat32, flagcxDevicePrimRecv};
  return flagcxSuccess;
}

static flagcxResult_t buildAllGatherGemmDag(
    flagcxUniRunnerState *runnerState, const void *input, const void *weight,
    void *output, float *scratch, size_t mPerRank, size_t n, size_t k,
    size_t slices, flagcxComm_t comm) {
  const int rank = comm->rank;
  const int nranks = comm->nranks;
  size_t gemmNodes = 0;
  size_t numNodes = 0;
  if (!checkedMul(static_cast<size_t>(nranks), slices, &gemmNodes) ||
      !checkedAdd(gemmNodes, nranks > 1 ? nranks - 1 : 0, &numNodes)) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(initializeDag(runnerState, numNodes));

  const int p2pCount = nranks > 1 ? nranks - 1 : 0;
  const int next = (rank + 1) % nranks;
  const int prev = (rank - 1 + nranks) % nranks;
  const size_t rankElements = mPerRank * k;
  const float *inputFloat = static_cast<const float *>(input);
  const float *weightFloat = static_cast<const float *>(weight);
  float *outputFloat = static_cast<float *>(output);

  for (int i = 0; i < p2pCount; ++i) {
    int txChunk = (rank - i + nranks) % nranks;
    int rxChunk = (rank - i - 1 + nranks) % nranks;
    void *send = i == 0 ? const_cast<void *>(input)
                        : static_cast<void *>(scratch + txChunk * rankElements);
    void *recv = static_cast<void *>(scratch + rxChunk * rankElements);
    FLAGCXCHECK(setP2pNode(&runnerState->dagNodes[i], i, i == 0 ? 0 : 1,
                           1 + (i < p2pCount - 1 ? 1 : 0), send, recv,
                           rankElements, next, prev));
  }

  for (int q = 0; q < nranks; ++q) {
    for (size_t s = 0; s < slices; ++s) {
      int nodeIdx = p2pCount + q * static_cast<int>(slices) + s;
      size_t kOffset = kSliceOffset(k, slices, s);
      const float *aBase = q == rank ? inputFloat : scratch + q * rankElements;
      setGemmNode(&runnerState->dagNodes[nodeIdx], nodeIdx, aBase + kOffset,
                  weightFloat + kOffset * n,
                  outputFloat + static_cast<size_t>(q) * mPerRank * n,
                  static_cast<uint32_t>(mPerRank), static_cast<uint32_t>(n),
                  static_cast<uint32_t>(kSliceLength(k, slices, s)),
                  static_cast<uint32_t>(k), static_cast<uint32_t>(n),
                  static_cast<uint32_t>(n), runnerState->uniRunnerNThreads,
                  s != 0, s == 0 ? (q == rank ? 0 : 1) : 1,
                  s + 1 < slices ? 1 : 0);
    }
  }

  for (int i = 0; i < p2pCount; ++i) {
    uniRunnerDagNode *node = &runnerState->dagNodes[i];
    FLAGCXCHECK(allocDagNodeDeps(node));
    int childSlot = 0;
    if (i > 0) {
      FLAGCXCHECK(setDagNodeParent(node, 0, i - 1));
    }
    if (i < p2pCount - 1) {
      node->children[childSlot++] = i + 1;
    }
    int rxChunk = (rank - i - 1 + nranks) % nranks;
    node->children[childSlot] =
        p2pCount + rxChunk * static_cast<int>(slices);
  }

  for (int q = 0; q < nranks; ++q) {
    for (size_t s = 0; s < slices; ++s) {
      int nodeIdx = p2pCount + q * static_cast<int>(slices) + s;
      uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
      FLAGCXCHECK(allocDagNodeDeps(node));
      if (s > 0) {
        FLAGCXCHECK(setDagNodeParent(node, 0, nodeIdx - 1));
      } else if (q != rank) {
        int receiveStep = (rank - q - 1 + nranks) % nranks;
        FLAGCXCHECK(setDagNodeParent(node, 0, receiveStep));
      }
      if (s + 1 < slices) {
        node->children[0] = nodeIdx + 1;
      }
    }
  }

  enqueueInitialNodes(runnerState);
  return validateDagNodes(runnerState);
}

static flagcxResult_t buildGemmReduceScatterDag(
    flagcxUniRunnerState *runnerState, const void *input, const void *weight,
    void *output, float *gemmOutput, float *rsScratch, size_t m, size_t n,
    size_t kPerRank, size_t slices, flagcxComm_t comm) {
  const int rank = comm->rank;
  const int nranks = comm->nranks;
  const size_t rowsPerRank = m / nranks;
  size_t gemmNodes = 0;
  size_t ringNodes = 0;
  size_t numNodes = 0;
  if (!checkedMul(static_cast<size_t>(nranks), slices, &gemmNodes) ||
      !checkedMul(nranks > 1 ? nranks - 1 : 0, 2, &ringNodes) ||
      !checkedAdd(gemmNodes, ringNodes, &numNodes)) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(initializeDag(runnerState, numNodes));

  const float *inputFloat = static_cast<const float *>(input);
  const float *weightFloat = static_cast<const float *>(weight);
  float *outputFloat = static_cast<float *>(output);
  const int ringBase = static_cast<int>(gemmNodes);
  const int next = (rank + 1) % nranks;
  const int prev = (rank - 1 + nranks) % nranks;
  const size_t chunkElements = rowsPerRank * n;

  for (int q = 0; q < nranks; ++q) {
    size_t rowStart = static_cast<size_t>(q) * rowsPerRank;
    for (size_t s = 0; s < slices; ++s) {
      int nodeIdx = q * static_cast<int>(slices) + s;
      size_t kOffset = kSliceOffset(kPerRank, slices, s);
      float *c = nranks == 1 ? outputFloat : gemmOutput + rowStart * n;
      setGemmNode(
          &runnerState->dagNodes[nodeIdx], nodeIdx,
          inputFloat + rowStart * kPerRank + kOffset,
          weightFloat + kOffset * n, c, static_cast<uint32_t>(rowsPerRank),
          static_cast<uint32_t>(n),
          static_cast<uint32_t>(kSliceLength(kPerRank, slices, s)),
          static_cast<uint32_t>(kPerRank), static_cast<uint32_t>(n),
          static_cast<uint32_t>(n), runnerState->uniRunnerNThreads, s != 0,
          s == 0 ? 0 : 1,
          s + 1 < slices ? 1 : (nranks > 1 ? 1 : 0));
    }
  }

  for (int i = 0; i < nranks - 1; ++i) {
    int txChunk = (rank - i - 1 + nranks) % nranks;
    int rxChunk = (rank - i - 2 + nranks) % nranks;
    int p2pIdx = ringBase + 2 * i;
    int redIdx = p2pIdx + 1;
    void *send = i == 0 ? static_cast<void *>(gemmOutput +
                                              txChunk * chunkElements)
                        : static_cast<void *>(rsScratch +
                                              txChunk * chunkElements);
    void *recv = static_cast<void *>(rsScratch + rxChunk * chunkElements);
    FLAGCXCHECK(setP2pNode(&runnerState->dagNodes[p2pIdx], p2pIdx, 2, 1,
                           send, recv, chunkElements, next, prev));

    uniRunnerDagNode *red = &runnerState->dagNodes[redIdx];
    red->nodeIdx = redIdx;
    red->nodeType = uniRunnerDagNodeTypeRed;
    red->numParents = 1;
    red->numChildren = i < nranks - 2 ? 1 : 0;
    red->nodeData.red.input1 = recv;
    red->nodeData.red.input2 = gemmOutput + rxChunk * chunkElements;
    red->nodeData.red.output = i == nranks - 2 ? output : recv;
    red->nodeData.red.count = chunkElements;
    red->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
    red->nodeData.red.datatype = flagcxFloat32;
    red->nodeData.red.redOp = flagcxSum;
    red->nodeData.red.triggerIdx = -1;
  }

  for (int q = 0; q < nranks; ++q) {
    for (size_t s = 0; s < slices; ++s) {
      int nodeIdx = q * static_cast<int>(slices) + s;
      uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
      FLAGCXCHECK(allocDagNodeDeps(node));
      if (s > 0) {
        FLAGCXCHECK(setDagNodeParent(node, 0, nodeIdx - 1));
      }
      if (s + 1 < slices) {
        node->children[0] = nodeIdx + 1;
      } else if (nranks > 1) {
        int firstTxChunk = (rank - 1 + nranks) % nranks;
        int step = q == firstTxChunk ? 0 : (rank - q - 2 + nranks) % nranks;
        node->children[0] = ringBase + 2 * step;
      }
    }
  }

  for (int i = 0; i < nranks - 1; ++i) {
    int txChunk = (rank - i - 1 + nranks) % nranks;
    int rxChunk = (rank - i - 2 + nranks) % nranks;
    int p2pIdx = ringBase + 2 * i;
    int redIdx = p2pIdx + 1;
    uniRunnerDagNode *p2p = &runnerState->dagNodes[p2pIdx];
    uniRunnerDagNode *red = &runnerState->dagNodes[redIdx];
    FLAGCXCHECK(allocDagNodeDeps(p2p));
    FLAGCXCHECK(allocDagNodeDeps(red));
    int firstParent = i == 0
                          ? txChunk * static_cast<int>(slices) + slices - 1
                          : redIdx - 2;
    int secondParent = rxChunk * static_cast<int>(slices) + slices - 1;
    FLAGCXCHECK(setDagNodeParent(p2p, 0, firstParent));
    FLAGCXCHECK(setDagNodeParent(p2p, 1, secondParent));
    p2p->children[0] = redIdx;
    FLAGCXCHECK(setDagNodeParent(red, 0, p2pIdx));
    if (i < nranks - 2) {
      red->children[0] = p2pIdx + 2;
    }
  }

  enqueueInitialNodes(runnerState);
  return validateDagNodes(runnerState);
}

static flagcxResult_t validateCommon(const void *input, const void *weight,
                                     void *output, size_t firstDim, size_t n,
                                     size_t k, flagcxDataType_t datatype,
                                     flagcxComm_t comm,
                                     flagcxStream_t stream) {
  if (input == NULL || weight == NULL || output == NULL || comm == NULL ||
      stream == NULL || firstDim == 0 || n == 0 || k == 0) {
    return flagcxInvalidArgument;
  }
  if (comm->nranks < 1) {
    return flagcxInvalidArgument;
  }
  if (comm->heteroComm == NULL || datatype != flagcxFloat32) {
    return flagcxNotSupported;
  }
  if (comm->heteroComm->proxyState == NULL) {
    return flagcxInvalidUsage;
  }
#ifndef COMPILE_KERNEL_HOST
  return flagcxNotSupported;
#else
  if (firstDim > UINT32_MAX || n > UINT32_MAX || k > UINT32_MAX) {
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
#endif
}

static flagcxResult_t finishInvocation(flagcxResult_t res, bool initialized,
                                       flagcxComm_t comm, void *workspace) {
  if (initialized) {
    flagcxResult_t cleanupRes = cleanupUniRunner(comm);
    if (res == flagcxSuccess) {
      res = cleanupRes;
    }
  }
  if (workspace != NULL) {
    flagcxResult_t freeRes =
        deviceAdaptor->deviceFree(workspace, flagcxMemDevice, NULL);
    if (res == flagcxSuccess) {
      res = freeRes;
    }
  }
  return res;
}

} // namespace

flagcxResult_t flagcxAllGatherGemm(
    const void *input, const void *weight, void *output, size_t mPerRank,
    size_t n, size_t k, flagcxDataType_t datatype, flagcxComm_t comm,
    flagcxStream_t stream) {
  flagcxResult_t res =
      validateCommon(input, weight, output, mPerRank, n, k, datatype, comm,
                     stream);
  if (res != flagcxSuccess) {
    return res;
  }
  if (flagcxParamUniRunnerFusedAgGemmAlgo() != 0) {
    return flagcxNotSupported;
  }

  size_t slices = 0;
  FLAGCXCHECK(getEffectiveSlices(k, &slices));
  size_t inputElements = 0, weightElements = 0, outputRows = 0;
  size_t outputElements = 0, inputBytes = 0, weightBytes = 0, outputBytes = 0;
  size_t workspaceElements = 0, workspaceBytes = 0;
  size_t gemmNodes = 0, numNodes = 0, dagBytes = 0;
  if (!checkedMul(mPerRank, k, &inputElements) ||
      !checkedMul(k, n, &weightElements) ||
      !checkedMul(static_cast<size_t>(comm->nranks), mPerRank, &outputRows) ||
      !checkedMul(outputRows, n, &outputElements) ||
      !checkedMul(inputElements, sizeof(float), &inputBytes) ||
      !checkedMul(weightElements, sizeof(float), &weightBytes) ||
      !checkedMul(outputElements, sizeof(float), &outputBytes) ||
      inputElements > UINT32_MAX ||
      !checkedMul(static_cast<size_t>(comm->nranks), inputElements,
                  &workspaceElements) ||
      !checkedMul(workspaceElements, sizeof(float), &workspaceBytes) ||
      !checkedMul(static_cast<size_t>(comm->nranks), slices, &gemmNodes) ||
      !checkedAdd(gemmNodes, comm->nranks > 1 ? comm->nranks - 1 : 0,
                  &numNodes) ||
      numNodes > static_cast<size_t>(INT_MAX) ||
      !checkedMul(numNodes, sizeof(uniRunnerDagNode), &dagBytes)) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(validateBufferRanges(input, inputBytes, weight, weightBytes,
                                   output, outputBytes));

  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
  void *workspace = NULL;
  if (comm->nranks > 1) {
    res = deviceAdaptor->deviceMalloc(&workspace, workspaceBytes,
                                      flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      return res;
    }
  }

  bool initialized = false;
  res = initUniRunner(comm, stream);
  if (res == flagcxSuccess) {
    initialized = true;
    flagcxUniRunnerState *runnerState =
        &comm->heteroComm->proxyState->uniRunnerState;
    if (runnerState->uniRunnerNThreads < 1 ||
        runnerState->uniRunnerNThreads > 1024 ||
        runnerState->uniRunnerNThreads > UINT16_MAX ||
        runnerState->uniRunnerNBlocks < 1) {
      res = flagcxInvalidArgument;
    } else {
      res = buildAllGatherGemmDag(
          runnerState, input, weight, output, static_cast<float *>(workspace),
          mPerRank, n, k, slices, comm);
      if (res == flagcxSuccess) {
        res = runUniRunner(comm);
      }
    }
  }
  return finishInvocation(res, initialized, comm, workspace);
}

flagcxResult_t flagcxGemmReduceScatter(
    const void *input, const void *weight, void *output, size_t m, size_t n,
    size_t kPerRank, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm, flagcxStream_t stream) {
  flagcxResult_t res = validateCommon(input, weight, output, m, n, kPerRank,
                                      datatype, comm, stream);
  if (res != flagcxSuccess) {
    return res;
  }
  if (op != flagcxSum) {
    return flagcxNotSupported;
  }
  if (m % static_cast<size_t>(comm->nranks) != 0) {
    return flagcxInvalidArgument;
  }
  if (flagcxParamUniRunnerFusedGemmRsAlgo() != 0) {
    return flagcxNotSupported;
  }

  size_t slices = 0;
  FLAGCXCHECK(getEffectiveSlices(kPerRank, &slices));
  size_t rowsPerRank = m / comm->nranks;
  size_t inputElements = 0, weightElements = 0, outputElements = 0;
  size_t fullOutputElements = 0, inputBytes = 0, weightBytes = 0;
  size_t outputBytes = 0, workspaceElements = 0, workspaceBytes = 0;
  size_t gemmNodes = 0, ringNodes = 0, numNodes = 0, dagBytes = 0;
  if (rowsPerRank > UINT32_MAX ||
      !checkedMul(m, kPerRank, &inputElements) ||
      !checkedMul(kPerRank, n, &weightElements) ||
      !checkedMul(rowsPerRank, n, &outputElements) ||
      !checkedMul(m, n, &fullOutputElements) || outputElements > UINT32_MAX ||
      !checkedMul(inputElements, sizeof(float), &inputBytes) ||
      !checkedMul(weightElements, sizeof(float), &weightBytes) ||
      !checkedMul(outputElements, sizeof(float), &outputBytes) ||
      !checkedMul(fullOutputElements, 2, &workspaceElements) ||
      !checkedMul(workspaceElements, sizeof(float), &workspaceBytes) ||
      !checkedMul(static_cast<size_t>(comm->nranks), slices, &gemmNodes) ||
      !checkedMul(comm->nranks > 1 ? comm->nranks - 1 : 0, 2, &ringNodes) ||
      !checkedAdd(gemmNodes, ringNodes, &numNodes) ||
      numNodes > static_cast<size_t>(INT_MAX) ||
      !checkedMul(numNodes, sizeof(uniRunnerDagNode), &dagBytes)) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(validateBufferRanges(input, inputBytes, weight, weightBytes,
                                   output, outputBytes));

  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
  void *workspace = NULL;
  if (comm->nranks > 1) {
    res = deviceAdaptor->deviceMalloc(&workspace, workspaceBytes,
                                      flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      return res;
    }
  }

  bool initialized = false;
  res = initUniRunner(comm, stream);
  if (res == flagcxSuccess) {
    initialized = true;
    flagcxUniRunnerState *runnerState =
        &comm->heteroComm->proxyState->uniRunnerState;
    if (runnerState->uniRunnerNThreads < 1 ||
        runnerState->uniRunnerNThreads > 1024 ||
        runnerState->uniRunnerNThreads > UINT16_MAX ||
        runnerState->uniRunnerNBlocks < 1) {
      res = flagcxInvalidArgument;
    } else {
      float *gemmOutput = static_cast<float *>(workspace);
      float *rsScratch =
          comm->nranks > 1 ? gemmOutput + fullOutputElements : NULL;
      res = buildGemmReduceScatterDag(
          runnerState, input, weight, output, gemmOutput, rsScratch, m, n,
          kPerRank, slices, comm);
      if (res == flagcxSuccess) {
        res = runUniRunner(comm);
      }
    }
  }
  return finishInvocation(res, initialized, comm, workspace);
}
