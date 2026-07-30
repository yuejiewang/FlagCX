/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "uni_runner_ascend.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#if defined(USE_ASCEND_ADAPTOR)
#include "ascend_adaptor.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "global_comm.h"
#include "param.h"

#include "hccl/hccl_comm.h"
#include "hccl/hccl_rank_graph.h"
#include "hccl/hccl_res.h"
#include "hccl/hcomm_primitives.h"

#if defined(__has_include)
#if __has_include("version/acl_extend_version.h")
#include "version/acl_extend_version.h"
#define FLAGCX_HAVE_ACL_EXTEND_VERSION 1
#endif
#endif

#include <cstring>
#include <mutex>
#include <new>
#include <vector>
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

#if defined(USE_ASCEND_ADAPTOR)

FLAGCX_PARAM(UniRunnerHccsChunkBytes, "UNIRUNNER_HCCS_CHUNK_BYTES",
             16 * 1024 * 1024);
FLAGCX_PARAM(UniRunnerHccsTimeoutMs, "UNIRUNNER_HCCS_TIMEOUT_MS", 30000);

namespace {

constexpr uint64_t kHccsBufferAlignment = 512;
constexpr uint32_t kHccsReadyNotify = 0;
constexpr uint32_t kHccsAckNotify = 1;
constexpr uint32_t kHccsChannelNotifyCount = 2;
constexpr uint32_t kHccsThreadNotifyCount = 1;
constexpr CommEngine kHccsEngine = CommEngine::COMM_ENGINE_CPU_TS;

struct AscendHccsState {
  std::mutex mutex;
  bool initAttempted = false;
  bool initialized = false;
  bool poisoned = false;
  flagcxResult_t initResult = flagcxSuccess;

  HcclComm hcclComm = nullptr;
  char commName[COMM_NAME_MAX_LENGTH] = {};
  int device = -1;
  uint32_t rank = 0;
  uint32_t nranks = 0;
  void *localBuffer = nullptr;
  uint64_t localBufferSize = 0;
  uint64_t slotBytes = 0;
  uint64_t chunkBytes = 0;
  uint32_t timeoutMs = 0;

  aclrtStream sendStream = nullptr;
  ThreadHandle sendThread = 0;
  std::vector<ChannelHandle> channels;
  std::vector<void *> remoteBuffers;
  std::vector<uint64_t> remoteBufferSizes;
  std::vector<void *> deferredScratch;

  // One platform-independent P2P DAG node is translated into one symmetric
  // HCOMM batch. Send/recv primitives register their bindings here; GroupEnd
  // validates the AlltoAll layout before invoking the HCCS data plane.
  bool groupActive = false;
  flagcxStream_t groupStream = nullptr;
  std::vector<const void *> groupSendBuffers;
  std::vector<void *> groupRecvBuffers;
  std::vector<size_t> groupSendCounts;
  std::vector<size_t> groupRecvCounts;
  std::vector<flagcxDataType_t> groupSendTypes;
  std::vector<flagcxDataType_t> groupRecvTypes;
  std::vector<bool> groupHasSend;
  std::vector<bool> groupHasRecv;
};

std::mutex hccsStateCreateMutex;

static flagcxResult_t hccsControlResult(HcclResult result, const char *op,
                                        uint32_t rank) {
  if (result == HCCL_SUCCESS)
    return flagcxSuccess;
  WARN("Ascend UniRunner HCCS rank %u: %s failed with HCCL status %d", rank,
       op, static_cast<int>(result));
  return result == HCCL_E_NOT_SUPPORT ? flagcxNotSupported
                                      : flagcxUnhandledDeviceError;
}

static flagcxResult_t hccsRuntimeResult(aclError result, const char *op,
                                        uint32_t rank) {
  if (result == ACL_SUCCESS)
    return flagcxSuccess;
  WARN("Ascend UniRunner HCCS rank %u: %s failed with ACL status %d", rank, op,
       static_cast<int>(result));
  return flagcxUnhandledDeviceError;
}

static flagcxResult_t agreeHccsSetupResult(flagcxComm_t comm,
                                           flagcxResult_t localResult) {
  if (comm == nullptr || comm->bootstrap == nullptr || comm->nranks <= 0)
    return flagcxInvalidArgument;

  const int32_t localCode = static_cast<int32_t>(localResult);
  std::vector<int32_t> allCodes(static_cast<size_t>(comm->nranks),
                                static_cast<int32_t>(flagcxInternalError));
  flagcxResult_t gatherResult =
      AllGatherBootstrap(comm->bootstrap, &localCode, allCodes.data(), 1,
                         flagcxInt32);
  if (gatherResult != flagcxSuccess)
    return gatherResult;

  // Select the first failing rank so every process returns the same status.
  for (int rank = 0; rank < comm->nranks; ++rank) {
    const int32_t code = allCodes[static_cast<size_t>(rank)];
    if (code == static_cast<int32_t>(flagcxSuccess))
      continue;
    if (code < 0 || code >= static_cast<int32_t>(flagcxNumResults))
      return flagcxInternalError;
    return static_cast<flagcxResult_t>(code);
  }
  return flagcxSuccess;
}

static flagcxResult_t validateHccsLinks(AscendHccsState *state) {
  uint32_t *layers = nullptr;
  uint32_t layerCount = 0;
  HcclResult result =
      HcclRankGraphGetLayers(state->hcclComm, &layers, &layerCount);
  flagcxResult_t converted =
      hccsControlResult(result, "HcclRankGraphGetLayers", state->rank);
  if (converted != flagcxSuccess)
    return converted;
  if (layerCount == 0 || layers == nullptr) {
    WARN("Ascend UniRunner HCCS rank %u: topology returned no network layers",
         state->rank);
    return flagcxNotSupported;
  }

  for (uint32_t peer = 0; peer < state->nranks; ++peer) {
    if (peer == state->rank)
      continue;

    bool foundHccs = false;
    for (uint32_t layerIndex = 0;
         layerIndex < layerCount && !foundHccs; ++layerIndex) {
      CommLink *links = nullptr;
      uint32_t linkCount = 0;
      result = HcclRankGraphGetLinks(state->hcclComm, layers[layerIndex],
                                     state->rank, peer, &links, &linkCount);
      converted =
          hccsControlResult(result, "HcclRankGraphGetLinks", state->rank);
      if (converted != flagcxSuccess)
        return converted;
      if (linkCount != 0 && links == nullptr) {
        WARN("Ascend UniRunner HCCS rank %u: topology returned %u links but "
             "a null link list for peer %u layer %u",
             state->rank, linkCount, peer, layers[layerIndex]);
        return flagcxInternalError;
      }
      for (uint32_t index = 0; index < linkCount; ++index) {
        if (links[index].linkAttr.linkProtocol !=
            CommProtocol::COMM_PROTOCOL_HCCS)
          continue;
        INFO(FLAGCX_UNIRUNNER,
             "Ascend UniRunner topology rank=%u peer=%u layer=%u "
             "protocol=HCCS hop=%u",
             state->rank, peer, layers[layerIndex],
             static_cast<unsigned>(links[index].linkAttr.hop));
        foundHccs = true;
        break;
      }
    }
    if (!foundHccs) {
      WARN("Ascend UniRunner HCCS rank %u: rank %u has no HCCS "
           "link; refusing transport fallback",
           state->rank, peer);
      return flagcxNotSupported;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t initializeHccsPreChannelState(AscendHccsState *state,
                                                    flagcxComm_t comm) {
  if (state == nullptr || comm == nullptr || comm->homoComm == nullptr ||
      comm->homoComm->base == nullptr || comm->heteroComm == nullptr)
    return flagcxInvalidArgument;

#if !defined(FLAGCX_HAVE_ACL_EXTEND_VERSION) ||                             \
    !defined(ACL_EXTEND_VERSION_STR)
  WARN("Ascend UniRunner HCCS AlltoAll requires CANN version metadata; "
       "refusing an unverified direct HCOMM API contract");
  return flagcxNotSupported;
#else
  // This CPU_TS batching sequence has been validated against the named CANN
  // release. Fail explicitly instead of silently using an incompatible HCOMM
  // submission contract on another release.
  if (std::strcmp(ACL_EXTEND_VERSION_STR, "9.0.0-beta.1") != 0) {
    WARN("Ascend UniRunner HCCS AlltoAll direct HCOMM path is validated only "
         "for CANN 9.0.0-beta.1 (build headers report %s); refusing an "
         "unverified submission mode",
         ACL_EXTEND_VERSION_STR);
    return flagcxNotSupported;
  }
#endif

  // This implementation deliberately targets one homogeneous, intra-server
  // HCCS island. A mismatched rank mapping would make AlltoAll block offsets
  // ambiguous, so reject it instead of silently selecting another path.
  if (comm->tuner != nullptr || comm->nranks < 2 ||
      comm->homoRanks != comm->nranks || comm->localRanks != comm->nranks ||
      comm->homoRank != comm->rank) {
    WARN("Ascend UniRunner HCCS AlltoAll requires one untuned, homogeneous "
         "intra-node communicator with identical global/HCCL rank ordering");
    return flagcxNotSupported;
  }

  state->device = comm->heteroComm->cudaDev;
  flagcxResult_t result =
      hccsRuntimeResult(aclrtSetDevice(state->device), "aclrtSetDevice",
                        static_cast<uint32_t>(comm->rank));
  if (result != flagcxSuccess)
    return result;

  state->hcclComm = comm->homoComm->base;
  HcclResult hcclResult =
      HcclGetCommName(state->hcclComm, state->commName);
  result = hccsControlResult(hcclResult, "HcclGetCommName", state->rank);
  if (result != flagcxSuccess)
    return result;
  hcclResult = HcclGetRankId(state->hcclComm, &state->rank);
  result = hccsControlResult(hcclResult, "HcclGetRankId", state->rank);
  if (result != flagcxSuccess)
    return result;
  hcclResult = HcclGetRankSize(state->hcclComm, &state->nranks);
  result = hccsControlResult(hcclResult, "HcclGetRankSize", state->rank);
  if (result != flagcxSuccess)
    return result;
  if (state->rank != static_cast<uint32_t>(comm->rank) ||
      state->nranks != static_cast<uint32_t>(comm->nranks)) {
    WARN("Ascend UniRunner HCCS rank mapping mismatch: HCCL=%u/%u, "
         "FlagCX=%d/%d",
         state->rank, state->nranks, comm->rank, comm->nranks);
    return flagcxNotSupported;
  }

  result = validateHccsLinks(state);
  if (result != flagcxSuccess)
    return result;

  hcclResult = HcclGetHcclBuffer(state->hcclComm, &state->localBuffer,
                                 &state->localBufferSize);
  result =
      hccsControlResult(hcclResult, "HcclGetHcclBuffer", state->rank);
  if (result != flagcxSuccess)
    return result;
  if (state->localBuffer == nullptr || state->localBufferSize < 4)
    return flagcxUnhandledDeviceError;

  result = hccsRuntimeResult(aclrtCreateStream(&state->sendStream),
                            "aclrtCreateStream(send)", state->rank);
  if (result != flagcxSuccess)
    return result;

  hcclResult = HcclThreadAcquireWithStream(
      state->hcclComm, kHccsEngine, state->sendStream, kHccsThreadNotifyCount,
      &state->sendThread);
  result = hccsControlResult(hcclResult, "HcclThreadAcquireWithStream(send)",
                            state->rank);
  if (result != flagcxSuccess)
    return result;

  return flagcxSuccess;
}

static flagcxResult_t initializeHccsState(AscendHccsState *state,
                                          flagcxComm_t comm) {
  // Channel creation may coordinate the two endpoints. Make every rank agree
  // that all purely local prerequisites succeeded before any rank enters
  // HcclChannelAcquire, otherwise one early local failure could strand a peer.
  flagcxResult_t result =
      initializeHccsPreChannelState(state, comm);
  result = agreeHccsSetupResult(comm, result);
  if (result != flagcxSuccess)
    return result;

  std::vector<uint32_t> peers;
  peers.reserve(state->nranks - 1);
  for (uint32_t peer = 0; peer < state->nranks; ++peer) {
    if (peer != state->rank)
      peers.push_back(peer);
  }
  std::vector<HcclChannelDesc> descriptions(peers.size());
  HcclResult hcclResult =
      HcclChannelDescInit(descriptions.data(),
                          static_cast<uint32_t>(descriptions.size()));
  result =
      hccsControlResult(hcclResult, "HcclChannelDescInit", state->rank);
  result = agreeHccsSetupResult(comm, result);
  if (result != flagcxSuccess)
    return result;
  for (size_t index = 0; index < peers.size(); ++index) {
    descriptions[index].remoteRank = peers[index];
    descriptions[index].channelProtocol =
        CommProtocol::COMM_PROTOCOL_HCCS;
    descriptions[index].notifyNum = kHccsChannelNotifyCount;
  }

  std::vector<ChannelHandle> acquiredChannels(peers.size());
  hcclResult =
      HcclChannelAcquire(state->hcclComm, kHccsEngine, descriptions.data(),
                         static_cast<uint32_t>(descriptions.size()),
                         acquiredChannels.data());
  result = hccsControlResult(hcclResult, "HcclChannelAcquire(HCCS)",
                            state->rank);

  state->channels.assign(state->nranks, 0);
  state->remoteBuffers.assign(state->nranks, nullptr);
  state->remoteBufferSizes.assign(state->nranks, 0);
  state->remoteBuffers[state->rank] = state->localBuffer;
  state->remoteBufferSizes[state->rank] = state->localBufferSize;

  // Two TX banks and two RX banks are used to put a completed HCOMM batch
  // between local staging and the HCCS write that consumes that staging.
  uint64_t smallestQuarterBuffer = state->localBufferSize / 4;
  // Acquire and GetHcclBuffer are one coordinated resource-creation phase.
  // A rank that returns first must keep progressing rather than entering a
  // Bootstrap rendezvous while its peer is still completing channel setup.
  if (result == flagcxSuccess) {
    for (size_t index = 0; index < peers.size(); ++index) {
      const uint32_t peer = peers[index];
      state->channels[peer] = acquiredChannels[index];
      hcclResult = HcclChannelGetHcclBuffer(
          state->hcclComm, state->channels[peer],
          &state->remoteBuffers[peer], &state->remoteBufferSizes[peer]);
      result = hccsControlResult(hcclResult, "HcclChannelGetHcclBuffer",
                                state->rank);
      if (result != flagcxSuccess)
        break;
      if (state->remoteBuffers[peer] == nullptr ||
          state->remoteBufferSizes[peer] < 4) {
        result = flagcxUnhandledDeviceError;
        break;
      }
      smallestQuarterBuffer = std::min(
          smallestQuarterBuffer, state->remoteBufferSizes[peer] / 4);
    }
  }
  result = agreeHccsSetupResult(comm, result);
  if (result != flagcxSuccess)
    return result;

  // Do not rely on every rank observing identical remote-buffer metadata.
  // Explicitly agree on the global minimum so all peers interpret slot
  // offsets in exactly the same way.
  std::vector<uint64_t> allQuarterBuffers(state->nranks, 0);
  result = AllGatherBootstrap(comm->bootstrap, &smallestQuarterBuffer,
                              allQuarterBuffers.data(), 1, flagcxUint64);
  if (result != flagcxSuccess)
    return result;
  const uint64_t globalSmallestQuarter =
      *std::min_element(allQuarterBuffers.begin(), allQuarterBuffers.end());
  state->slotBytes = globalSmallestQuarter / state->nranks;
  state->slotBytes -= state->slotBytes % kHccsBufferAlignment;
  const int64_t configuredChunk = flagcxParamUniRunnerHccsChunkBytes();
  const int64_t configuredTimeout = flagcxParamUniRunnerHccsTimeoutMs();
  if (state->slotBytes == 0 || configuredChunk <= 0 ||
      configuredTimeout <= 0 ||
      static_cast<uint64_t>(configuredTimeout) >
          std::numeric_limits<uint32_t>::max()) {
    WARN("Ascend UniRunner HCCS has invalid staging/timeout configuration");
    return flagcxInvalidArgument;
  }
  state->chunkBytes =
      std::min(state->slotBytes, static_cast<uint64_t>(configuredChunk));
  if (state->chunkBytes >= kHccsBufferAlignment)
    state->chunkBytes -= state->chunkBytes % kHccsBufferAlignment;
  if (state->chunkBytes == 0)
    state->chunkBytes = std::min(state->slotBytes,
                                 static_cast<uint64_t>(configuredChunk));
  state->timeoutMs = static_cast<uint32_t>(configuredTimeout);

  INFO(FLAGCX_UNIRUNNER,
       "Ascend UniRunner HCCS initialized: rank=%u nranks=%u engine=CPU_TS "
       "protocol=HCCS hccl_delegate=0 socket=0 host_staging=0 "
       "buffer_mode=HCCL_STAGED_PING_PONG banks=2 slot=%llu chunk=%llu "
       "timeout_ms=%u",
       state->rank, state->nranks,
       static_cast<unsigned long long>(state->slotBytes),
       static_cast<unsigned long long>(state->chunkBytes), state->timeoutMs);
  return flagcxSuccess;
}

static AscendHccsState *getOrCreateHccsState(flagcxComm_t comm) {
  std::lock_guard<std::mutex> guard(hccsStateCreateMutex);
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr) {
    state = new (std::nothrow) AscendHccsState();
    if (state != nullptr)
      comm->ascendHccsState = state;
  }
  return state;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsPrepare(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  AscendHccsState *state = getOrCreateHccsState(comm);
  if (state == nullptr)
    return agreeHccsSetupResult(comm, flagcxSystemError);

  std::lock_guard<std::mutex> guard(state->mutex);
  if (!state->initAttempted) {
    // Allocation is rank-local. Agree before entering coordinated channel
    // acquisition so a failed rank cannot strand its peers.
    flagcxResult_t allocationResult =
        agreeHccsSetupResult(comm, flagcxSuccess);
    if (allocationResult != flagcxSuccess) {
      state->initAttempted = true;
      state->initResult = allocationResult;
      state->poisoned = true;
      return allocationResult;
    }
    state->initAttempted = true;
    flagcxResult_t localResult = initializeHccsState(state, comm);
    state->initResult = agreeHccsSetupResult(comm, localResult);
    state->initialized = state->initResult == flagcxSuccess;
    state->poisoned = !state->initialized;
  }
  if (!state->initialized || state->poisoned)
    return state->initResult == flagcxSuccess ? flagcxUnhandledDeviceError
                                              : state->initResult;
  return flagcxSuccess;
}

static void resetHccsP2pGroup(AscendHccsState *state) {
  state->groupActive = false;
  state->groupStream = nullptr;
  state->groupSendBuffers.clear();
  state->groupRecvBuffers.clear();
  state->groupSendCounts.clear();
  state->groupRecvCounts.clear();
  state->groupSendTypes.clear();
  state->groupRecvTypes.clear();
  state->groupHasSend.clear();
  state->groupHasRecv.clear();
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsGroupStart(flagcxComm_t comm,
                                   flagcxStream_t stream) {
  if (comm == nullptr || stream == nullptr || comm->nranks < 2)
    return flagcxInvalidArgument;
  FLAGCXCHECK(flagcxAscendUniRunnerHccsPrepare(comm));
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxInternalError;

  std::lock_guard<std::mutex> guard(state->mutex);
  if (state->groupActive)
    return flagcxInvalidUsage;
  const size_t ranks = static_cast<size_t>(comm->nranks);
  state->groupActive = true;
  state->groupStream = stream;
  state->groupSendBuffers.assign(ranks, nullptr);
  state->groupRecvBuffers.assign(ranks, nullptr);
  state->groupSendCounts.assign(ranks, 0);
  state->groupRecvCounts.assign(ranks, 0);
  state->groupSendTypes.assign(ranks, flagcxInt8);
  state->groupRecvTypes.assign(ranks, flagcxInt8);
  state->groupHasSend.assign(ranks, false);
  state->groupHasRecv.assign(ranks, false);
  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsSend(
    const void *sendbuff, size_t count, flagcxDataType_t datatype, int peer,
    flagcxComm_t comm) {
  if (comm == nullptr || peer < 0 || peer >= comm->nranks ||
      (count != 0 && sendbuff == nullptr))
    return flagcxInvalidArgument;
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxInvalidUsage;
  std::lock_guard<std::mutex> guard(state->mutex);
  const size_t index = static_cast<size_t>(peer);
  if (!state->groupActive || state->groupHasSend[index])
    return flagcxInvalidUsage;
  state->groupSendBuffers[index] = sendbuff;
  state->groupSendCounts[index] = count;
  state->groupSendTypes[index] = datatype;
  state->groupHasSend[index] = true;
  return flagcxSuccess;
}

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsRecv(
    void *recvbuff, size_t count, flagcxDataType_t datatype, int peer,
    flagcxComm_t comm) {
  if (comm == nullptr || peer < 0 || peer >= comm->nranks ||
      (count != 0 && recvbuff == nullptr))
    return flagcxInvalidArgument;
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxInvalidUsage;
  std::lock_guard<std::mutex> guard(state->mutex);
  const size_t index = static_cast<size_t>(peer);
  if (!state->groupActive || state->groupHasRecv[index])
    return flagcxInvalidUsage;
  state->groupRecvBuffers[index] = recvbuff;
  state->groupRecvCounts[index] = count;
  state->groupRecvTypes[index] = datatype;
  state->groupHasRecv[index] = true;
  return flagcxSuccess;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsGroupEnd(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxInvalidUsage;

  const void *sendBase = nullptr;
  void *recvBase = nullptr;
  size_t count = 0;
  flagcxDataType_t datatype = flagcxInt8;
  flagcxStream_t stream = nullptr;
  flagcxResult_t validation = flagcxSuccess;
  {
    std::lock_guard<std::mutex> guard(state->mutex);
    if (!state->groupActive)
      return flagcxInvalidUsage;
    stream = state->groupStream;
    if (state->groupHasSend.empty() || state->groupHasRecv.empty()) {
      validation = flagcxInvalidUsage;
    } else {
      count = state->groupSendCounts[0];
      datatype = state->groupSendTypes[0];
      const size_t typeSize = getFlagcxDataTypeSize(datatype);
      if (typeSize == 0 ||
          count > std::numeric_limits<size_t>::max() / typeSize) {
        validation = flagcxInvalidArgument;
      } else {
        const size_t peerBytes = count * typeSize;
        sendBase = state->groupSendBuffers[0];
        recvBase = state->groupRecvBuffers[0];
        const uintptr_t sendAddress =
            reinterpret_cast<uintptr_t>(sendBase);
        const uintptr_t recvAddress =
            reinterpret_cast<uintptr_t>(recvBase);
        for (int peer = 0; peer < comm->nranks; ++peer) {
          const size_t index = static_cast<size_t>(peer);
          if (!state->groupHasSend[index] || !state->groupHasRecv[index] ||
              state->groupSendCounts[index] != count ||
              state->groupRecvCounts[index] != count ||
              state->groupSendTypes[index] != datatype ||
              state->groupRecvTypes[index] != datatype ||
              reinterpret_cast<uintptr_t>(state->groupSendBuffers[index]) !=
                  sendAddress + index * peerBytes ||
              reinterpret_cast<uintptr_t>(state->groupRecvBuffers[index]) !=
                  recvAddress + index * peerBytes) {
            validation = flagcxNotSupported;
            break;
          }
        }
      }
    }
    resetHccsP2pGroup(state);
  }
  if (validation != flagcxSuccess)
    return validation;
  return flagcxAscendUniRunnerHccsAlltoAll(
      sendBase, recvBase, count, datatype, comm, stream);
}

static int checkedHcommResult(int result, const char *operation,
                              uint32_t rank) {
  if (result != 0) {
    WARN("Ascend UniRunner HCCS rank %u: %s failed with HCOMM status %d",
         rank, operation, result);
  }
  return result;
}

static uint8_t *hccsTxSlot(AscendHccsState *state, uint32_t bank,
                           uint32_t peer) {
  uint8_t *base = static_cast<uint8_t *>(state->localBuffer);
  return base + (static_cast<uint64_t>(bank) * state->nranks + peer) *
                    state->slotBytes;
}

static uint8_t *hccsRxSlot(AscendHccsState *state, void *buffer,
                           uint32_t bank, uint32_t sourceRank) {
  uint8_t *base = static_cast<uint8_t *>(buffer);
  return base +
         ((2ULL + bank) * state->nranks + sourceRank) * state->slotBytes;
}

static flagcxResult_t submitHccsPipelinePhase(
    AscendHccsState *state, flagcxComm_t comm, const uint8_t *source,
    size_t peerBytes, size_t offset, uint64_t bytes, uint32_t stageBank,
    uint32_t writeBank, bool *drained) {
  *drained = false;
  int firstError = 0;

  // Atlas A2 executes all HCOMM data-plane primitives as a batch. The only
  // cross-device data movement below is HcommWrite on an HCCS channel. The
  // current input is staged into one bank while a bank completed by the
  // previous batch is consumed by HcommWrite. This explicit batch boundary
  // avoids the CPU+TS scheduler reading the previous contents of a slot.
  const bool batchStarted =
      checkedHcommResult(HcommBatchModeStart(""),
                         "HcommBatchModeStart", state->rank) == 0;
  if (!batchStarted) {
    firstError = 1;
  } else {
    for (uint32_t peer = 0; peer < state->nranks && firstError == 0; ++peer) {
      if (peer == state->rank)
        continue;
      firstError = checkedHcommResult(
          HcommLocalCopyOnThread(
              state->sendThread, hccsTxSlot(state, stageBank, peer),
              source + static_cast<size_t>(peer) * peerBytes + offset, bytes),
          "HcommLocalCopyOnThread(stage)", state->rank);
    }

    // Publish readiness for every outbound peer block.
    for (uint32_t peer = 0; peer < state->nranks && firstError == 0; ++peer) {
      if (peer == state->rank)
        continue;
      firstError = checkedHcommResult(
          HcommChannelNotifyRecordOnThread(state->sendThread,
                                           state->channels[peer],
                                           kHccsReadyNotify),
          "HcommChannelNotifyRecordOnThread(ready)", state->rank);
    }

    // Do not start writes until all endpoints have submitted the same phase.
    for (uint32_t peer = 0; peer < state->nranks && firstError == 0; ++peer) {
      if (peer == state->rank)
        continue;
      firstError = checkedHcommResult(
          HcommChannelNotifyWaitOnThread(state->sendThread,
                                         state->channels[peer],
                                         kHccsReadyNotify,
                                         state->timeoutMs),
          "HcommChannelNotifyWaitOnThread(ready)", state->rank);
    }

    // HCCS one-sided writes consume the bank completed by the preceding
    // batch and populate the matching RX bank on the peer.
    for (uint32_t peer = 0; peer < state->nranks && firstError == 0; ++peer) {
      if (peer == state->rank)
        continue;
      firstError = checkedHcommResult(
          HcommWriteOnThread(state->sendThread, state->channels[peer],
                             hccsRxSlot(state, state->remoteBuffers[peer],
                                        writeBank, state->rank),
                             hccsTxSlot(state, writeBank, peer), bytes),
          "HcommWriteOnThread", state->rank);
      if (firstError == 0)
        firstError = checkedHcommResult(
            HcommChannelNotifyRecordOnThread(state->sendThread,
                                             state->channels[peer],
                                             kHccsAckNotify),
            "HcommChannelNotifyRecordOnThread(done)", state->rank);
    }

    // Wait until every peer has published completion of its reciprocal write.
    for (uint32_t peer = 0; peer < state->nranks && firstError == 0; ++peer) {
      if (peer == state->rank)
        continue;
      firstError = checkedHcommResult(
          HcommChannelNotifyWaitOnThread(state->sendThread,
                                         state->channels[peer],
                                         kHccsAckNotify,
                                         state->timeoutMs),
          "HcommChannelNotifyWaitOnThread(done)", state->rank);
    }

    const int endError =
        checkedHcommResult(HcommBatchModeEnd(""),
                           "HcommBatchModeEnd", state->rank);
    if (firstError == 0)
      firstError = endError;
  }

  int sendSync = 0;
  if (batchStarted) {
    sendSync = checkedHcommResult(
        HcommThreadSynchronize(state->sendThread),
        "HcommThreadSynchronize(exchange)", state->rank);
    if (sendSync == 0) {
      const flagcxResult_t streamResult = hccsRuntimeResult(
          aclrtSynchronizeStream(state->sendStream),
          "aclrtSynchronizeStream(HCOMM exchange)", state->rank);
      if (streamResult != flagcxSuccess)
        sendSync = 1;
    }
    if (sendSync == 0) {
      const flagcxResult_t deviceResult = hccsRuntimeResult(
          aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(HCOMM exchange)",
          state->rank);
      if (deviceResult != flagcxSuccess)
        sendSync = 1;
    }
    if (firstError == 0)
      firstError = sendSync;
  }
  *drained = batchStarted && sendSync == 0;

  // Synchronize rank control planes only after the complete HCOMM batch is
  // drained, before either rank advances to the next ping-pong phase.
  return agreeHccsSetupResult(
      comm, firstError == 0 ? flagcxSuccess : flagcxUnhandledDeviceError);
}

static flagcxResult_t submitHccsChunkRound(
    AscendHccsState *state, flagcxComm_t comm, const uint8_t *source,
    uint8_t *destination, size_t peerBytes, size_t offset, uint64_t bytes,
    bool *drained) {
  *drained = false;

  // Phase 0 primes TX bank 0. Its write consumes the old contents of bank 1
  // into RX bank 1, which is deliberately ignored.
  bool primeDrained = false;
  flagcxResult_t result = submitHccsPipelinePhase(
      state, comm, source, peerBytes, offset, bytes, 0, 1, &primeDrained);
  if (result != flagcxSuccess)
    return result;

  // Phase 1 stages bank 1 while writing the now-current bank 0 into RX bank
  // 0. Both phases have an identical, complete HCOMM graph; unlike a
  // write-only second batch, this is accepted by the A2 CPU+TS executor.
  bool exchangeDrained = false;
  result = submitHccsPipelinePhase(
      state, comm, source, peerBytes, offset, bytes, 1, 0, &exchangeDrained);
  *drained = primeDrained && exchangeDrained;
  if (result != flagcxSuccess)
    return result;

  // The cross-device movement above is HCOMM/HCCS. This final copy is purely
  // local D2D movement from the completed RX bank into user output.
  flagcxResult_t copyResult = flagcxSuccess;
  for (uint32_t peer = 0; peer < state->nranks; ++peer) {
    if (peer == state->rank)
      continue;
    copyResult = hccsRuntimeResult(
        aclrtMemcpy(destination + static_cast<size_t>(peer) * peerBytes +
                        offset,
                    bytes,
                    hccsRxSlot(state, state->localBuffer, 0, peer), bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE),
        "aclrtMemcpy(HCCS RX to output)", state->rank);
    if (copyResult != flagcxSuccess)
      break;
  }
  if (copyResult == flagcxSuccess) {
    // aclrtMemcpy(D2D) may return before the copy engine has stopped reading
    // the HCCL RX slot. Drain it before the next chunk reuses that slot.
    copyResult = hccsRuntimeResult(
        aclrtSynchronizeDevice(), "aclrtSynchronizeDevice(output copy)",
        state->rank);
  }
  return agreeHccsSetupResult(comm, copyResult);
}

static flagcxResult_t freeOrDeferHccsScratch(AscendHccsState *state,
                                             void *scratch, bool drained) {
  if (scratch == nullptr)
    return flagcxSuccess;
  if (!drained) {
    state->deferredScratch.push_back(scratch);
    WARN("Ascend UniRunner HCCS rank %u: retaining in-place snapshot %p "
         "until communicator teardown because stream drain failed",
         state->rank, scratch);
    return flagcxUnhandledDeviceError;
  }
  return hccsRuntimeResult(aclrtFree(scratch), "aclrtFree(in-place snapshot)",
                           state->rank);
}

} // namespace

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsAlltoAll(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxComm_t comm, flagcxStream_t stream) {
  if (comm == nullptr || stream == nullptr)
    return flagcxInvalidArgument;

  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0 || count > std::numeric_limits<size_t>::max() / typeSize)
    return flagcxInvalidArgument;
  const size_t peerBytes = count * typeSize;
  if (peerBytes == 0)
    return flagcxSuccess;
  if (sendbuff == nullptr || recvbuff == nullptr || comm->nranks <= 0 ||
      peerBytes >
          std::numeric_limits<size_t>::max() /
              static_cast<size_t>(comm->nranks))
    return flagcxInvalidArgument;
  const size_t totalBytes = peerBytes * static_cast<size_t>(comm->nranks);

  FLAGCXCHECK(flagcxAscendUniRunnerHccsPrepare(comm));
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxInternalError;

  std::lock_guard<std::mutex> guard(state->mutex);

  flagcxResult_t result =
      hccsRuntimeResult(aclrtSetDevice(state->device), "aclrtSetDevice",
                        state->rank);
  if (result != flagcxSuccess) {
    state->poisoned = true;
    return result;
  }

  INFO(FLAGCX_UNIRUNNER,
       "Ascend UniRunner AlltoAll selected=HCOMM_HCCS runner=UNIRUNNER "
       "rank=%u peer_bytes=%zu hccl_collective=0 socket=0 host_staging=0",
       state->rank, peerBytes);

  // This implementation is synchronous by design. Ordering after the caller
  // stream here also makes a full in-place snapshot safe before any receive
  // overwrites the user's send blocks.
  result = hccsRuntimeResult(aclrtSynchronizeStream(stream->base),
                            "aclrtSynchronizeStream(caller)", state->rank);
  const bool inPlace = sendbuff == recvbuff;

  const void *effectiveSend = sendbuff;
  void *snapshot = nullptr;
  if (inPlace && result == flagcxSuccess) {
    result = hccsRuntimeResult(
        aclrtMalloc(&snapshot, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(in-place snapshot)", state->rank);
    if (result == flagcxSuccess) {
      result = hccsRuntimeResult(
          aclrtMemcpy(snapshot, totalBytes, sendbuff, totalBytes,
                      ACL_MEMCPY_DEVICE_TO_DEVICE),
          "aclrtMemcpy(in-place snapshot)", state->rank);
    }
    if (result == flagcxSuccess) {
      effectiveSend = snapshot;
    } else if (snapshot != nullptr) {
      (void)aclrtFree(snapshot);
      snapshot = nullptr;
    }
  }

  const uint8_t *source = static_cast<const uint8_t *>(effectiveSend);
  uint8_t *destination = static_cast<uint8_t *>(recvbuff);

  // The self block does not traverse a link, but preserving it is part of the
  // AlltoAll contract. Cross-rank blocks below exclusively use HCOMM/HCCS.
  if (result == flagcxSuccess &&
      source + state->rank * peerBytes !=
      destination + state->rank * peerBytes) {
    result = hccsRuntimeResult(
        aclrtMemcpy(destination + state->rank * peerBytes, peerBytes,
                    source + state->rank * peerBytes, peerBytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE),
        "aclrtMemcpy(self block)", state->rank);
  }

  // Every rank must agree that stream ordering, optional snapshot creation,
  // and the self copy succeeded before any rank enters the channel protocol.
  // Otherwise one local failure could strand its peers in a notify wait.
  flagcxResult_t preflightResult = agreeHccsSetupResult(comm, result);
  if (preflightResult != flagcxSuccess) {
    if (snapshot != nullptr)
      (void)aclrtFree(snapshot);
    state->poisoned = true;
    return preflightResult;
  }

  bool allWorkDrained = true;
  bool hcommCommAcquired = false;
  if (result == flagcxSuccess) {
    const int acquireResult = checkedHcommResult(
        HcommAcquireComm(state->commName), "HcommAcquireComm", state->rank);
    if (acquireResult == 0) {
      hcommCommAcquired = true;
    } else {
      result = flagcxUnhandledDeviceError;
    }
  }
  result = agreeHccsSetupResult(comm, result);
  if (result == flagcxSuccess) {
    for (size_t offset = 0; offset < peerBytes;
         offset += static_cast<size_t>(state->chunkBytes)) {
      const uint64_t bytes = static_cast<uint64_t>(
          std::min(peerBytes - offset,
                   static_cast<size_t>(state->chunkBytes)));
      bool chunkDrained = true;
      result = submitHccsChunkRound(state, comm, source, destination,
                                    peerBytes, offset, bytes, &chunkDrained);
      allWorkDrained = allWorkDrained && chunkDrained;
      if (result != flagcxSuccess) {
        WARN("Ascend UniRunner HCCS AlltoAll rank %u failed at offset=%zu "
             "bytes=%llu; communicator is poisoned",
             state->rank, offset, static_cast<unsigned long long>(bytes));
        break;
      }
    }
  }
  if (hcommCommAcquired) {
    const int releaseResult = checkedHcommResult(
        HcommReleaseComm(state->commName), "HcommReleaseComm", state->rank);
    if (result == flagcxSuccess && releaseResult != 0)
      result = flagcxUnhandledDeviceError;
  }
  result = agreeHccsSetupResult(comm, result);

  flagcxResult_t snapshotResult =
      freeOrDeferHccsScratch(state, snapshot, allWorkDrained);
  if (result == flagcxSuccess)
    result = snapshotResult;
  result = agreeHccsSetupResult(comm, result);
  if (result != flagcxSuccess)
    state->poisoned = true;
  return result;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsPrepareDestroy(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxSuccess;

  std::lock_guard<std::mutex> guard(state->mutex);
  if (state->device >= 0 &&
      aclrtSetDevice(state->device) != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  aclError sendResult = ACL_SUCCESS;
  if (state->sendStream != nullptr)
    sendResult = aclrtSynchronizeStream(state->sendStream);
  if (sendResult != ACL_SUCCESS)
    (void)hccsRuntimeResult(sendResult,
                            "aclrtSynchronizeStream(send teardown)",
                            state->rank);
  return sendResult == ACL_SUCCESS ? flagcxSuccess
                                   : flagcxUnhandledDeviceError;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsFinishDestroy(flagcxComm_t comm) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  AscendHccsState *state =
      static_cast<AscendHccsState *>(comm->ascendHccsState);
  if (state == nullptr)
    return flagcxSuccess;

  flagcxResult_t result = flagcxSuccess;
  if (state->device >= 0 &&
      aclrtSetDevice(state->device) != ACL_SUCCESS)
    result = flagcxUnhandledDeviceError;
  if (state->sendStream != nullptr &&
      aclrtDestroyStream(state->sendStream) != ACL_SUCCESS)
    result = flagcxUnhandledDeviceError;
  for (void *scratch : state->deferredScratch) {
    if (aclrtFree(scratch) != ACL_SUCCESS)
      result = flagcxUnhandledDeviceError;
  }
  delete state;
  comm->ascendHccsState = nullptr;
  return result;
}

#else

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsPrepare(flagcxComm_t comm) {
  (void)comm;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsGroupStart(flagcxComm_t comm,
                                   flagcxStream_t stream) {
  (void)comm;
  (void)stream;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsSend(
    const void *sendbuff, size_t count, flagcxDataType_t datatype, int peer,
    flagcxComm_t comm) {
  (void)sendbuff;
  (void)count;
  (void)datatype;
  (void)peer;
  (void)comm;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsRecv(
    void *recvbuff, size_t count, flagcxDataType_t datatype, int peer,
    flagcxComm_t comm) {
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)peer;
  (void)comm;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsGroupEnd(flagcxComm_t comm) {
  (void)comm;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t flagcxAscendUniRunnerHccsAlltoAll(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxComm_t comm, flagcxStream_t stream) {
  (void)sendbuff;
  (void)recvbuff;
  (void)count;
  (void)datatype;
  (void)comm;
  (void)stream;
  return flagcxNotSupported;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsPrepareDestroy(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

extern "C" flagcxResult_t
flagcxAscendUniRunnerHccsFinishDestroy(flagcxComm_t comm) {
  (void)comm;
  return flagcxSuccess;
}

#endif
