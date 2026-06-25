#include "uni_runner_impl.h"
#include "adaptor.h"
#include "bootstrap.h"
#include "comm.h"
#include "device_api/flagcx_device.h"
#include "flagcx_hetero.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
#include "socket.h"
#include "transport.h"
#include "uni_runner_helper.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdint>
#include <math.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

FLAGCX_PARAM(UniRunnerNSlices, "UNIRUNNER_NSLICES", 1);
FLAGCX_PARAM(UniRunnerNThreads, "UNIRUNNER_NTHREADS", 32);
FLAGCX_PARAM(UniRunnerNBlocks, "UNIRUNNER_NBLOCKS", 1);
FLAGCX_PARAM(UniRunnerNRedSlices, "UNIRUNNER_NREDSLICES", 0);
FLAGCX_PARAM(UniRunnerRedSliceSize, "UNIRUNNER_REDSLICESIZE", 65536);

namespace {

constexpr char kUniRunnerDagCachePathEnv[] = "FLAGCX_UNIRUNNER_DAG_CACHE_PATH";

struct uniRunnerDagRuntimeBindings {
  const void *inputBase = NULL;
  void *outputBase = NULL;
  void *scratchBase = NULL;
  void *flagBase = NULL;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t scratchBytes = 0;
  size_t flagBytes = 0;
};

std::mutex gUniRunnerDagCacheMutex;
std::unordered_map<size_t, std::vector<std::shared_ptr<uniRunnerDagTemplate>>>
    gUniRunnerDagCache;
std::unordered_set<std::string> gUniRunnerDagLoadedPaths;

static size_t hashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

static bool uniRunnerDagCacheKeysEqual(const uniRunnerDagCacheKey &lhs,
                                       const uniRunnerDagCacheKey &rhs) {
  return lhs.algoType == rhs.algoType && lhs.commOp == rhs.commOp &&
         lhs.count == rhs.count && lhs.datatype == rhs.datatype &&
         lhs.redOp == rhs.redOp && lhs.rank == rhs.rank &&
         lhs.nranks == rhs.nranks && lhs.root == rhs.root &&
         lhs.groupSize == rhs.groupSize && lhs.numSlices == rhs.numSlices &&
         lhs.numRedSlices == rhs.numRedSlices &&
         lhs.redSliceSize == rhs.redSliceSize && lhs.nthreads == rhs.nthreads &&
         lhs.inputOutputAliased == rhs.inputOutputAliased &&
         lhs.inputScratchAliased == rhs.inputScratchAliased &&
         lhs.outputScratchAliased == rhs.outputScratchAliased;
}

static flagcxResult_t insertUniRunnerDagTemplateLocked(
    const std::shared_ptr<uniRunnerDagTemplate> &dagTemplate) {
  std::vector<std::shared_ptr<uniRunnerDagTemplate>> &bucket =
      gUniRunnerDagCache[dagTemplate->hashValue];
  for (std::shared_ptr<uniRunnerDagTemplate> &entry : bucket) {
    if (uniRunnerDagCacheKeysEqual(entry->key, dagTemplate->key)) {
      entry = dagTemplate;
      return flagcxSuccess;
    }
  }
  bucket.push_back(dagTemplate);
  return flagcxSuccess;
}

static bool findUniRunnerDagTemplateLocked(
    size_t hashValue, const uniRunnerDagCacheKey &key,
    std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::unordered_map<
      size_t, std::vector<std::shared_ptr<uniRunnerDagTemplate>>>::iterator it =
      gUniRunnerDagCache.find(hashValue);
  if (it == gUniRunnerDagCache.end()) {
    return false;
  }
  for (const std::shared_ptr<uniRunnerDagTemplate> &entry : it->second) {
    if (uniRunnerDagCacheKeysEqual(entry->key, key)) {
      *dagTemplate = entry;
      return true;
    }
  }
  return false;
}

static flagcxResult_t importUniRunnerDagTemplatesLocked(
    const std::vector<uniRunnerDagTemplate> &dagTemplates) {
  for (const uniRunnerDagTemplate &dagTemplate : dagTemplates) {
    FLAGCXCHECK(insertUniRunnerDagTemplateLocked(
        std::make_shared<uniRunnerDagTemplate>(dagTemplate)));
  }
  return flagcxSuccess;
}

static std::string getUniRunnerDagEnvPath(const char *envName) {
  const char *path = flagcxGetEnv(envName);
  return path == NULL ? "" : std::string(path);
}

static std::string getUniRunnerDagCacheDirFromEnv() {
  return getUniRunnerDagEnvPath(kUniRunnerDagCachePathEnv);
}

static std::string makeUniRunnerDagFileName(size_t hashValue, int rank) {
  return "dag_hash_" + std::to_string(hashValue) + "_rank_" +
         std::to_string(rank) + ".json";
}

static std::string joinUniRunnerDagDirAndFile(const std::string &dir,
                                              const std::string &fileName) {
  if (dir.empty()) {
    return "";
  }
  if (dir.back() == '/') {
    return dir + fileName;
  }
  return dir + "/" + fileName;
}

static std::string makeUniRunnerDagFilePath(const std::string &dir,
                                            size_t hashValue, int rank) {
  return joinUniRunnerDagDirAndFile(dir,
                                    makeUniRunnerDagFileName(hashValue, rank));
}

static flagcxResult_t ensureUniRunnerDagDirExists(const std::string &dir) {
  if (dir.empty()) {
    return flagcxSuccess;
  }

  std::string current;
  if (dir[0] == '/') {
    current = "/";
  }

  size_t pos = 0;
  while (pos < dir.size()) {
    size_t next = dir.find('/', pos);
    std::string segment = dir.substr(
        pos, next == std::string::npos ? std::string::npos : next - pos);
    if (!segment.empty()) {
      if (!current.empty() && current.back() != '/') {
        current += "/";
      }
      current += segment;

      struct stat st;
      if (stat(current.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
          return flagcxSystemError;
        }
      } else if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        return flagcxSystemError;
      }
    }
    if (next == std::string::npos) {
      break;
    }
    pos = next + 1;
  }
  return flagcxSuccess;
}

static flagcxResult_t loadUniRunnerDagFileIntoCache(const std::string &path) {
  if (path.empty()) {
    return flagcxSuccess;
  }

  {
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    if (gUniRunnerDagLoadedPaths.count(path) != 0) {
      return flagcxSuccess;
    }
  }

  std::vector<uniRunnerDagTemplate> dagTemplates;
  FLAGCXCHECK(uniRunnerLoadDagJsonFileIfExists(path, &dagTemplates));
  if (dagTemplates.empty()) {
    return flagcxSuccess;
  }

  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  if (gUniRunnerDagLoadedPaths.count(path) != 0) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(importUniRunnerDagTemplatesLocked(dagTemplates));
  gUniRunnerDagLoadedPaths.insert(path);
  return flagcxSuccess;
}

static bool matchBindingRange(const void *ptr, const void *base, size_t bytes,
                              int64_t *offsetBytes) {
  if (ptr == NULL || base == NULL) {
    return false;
  }
  uintptr_t ptrAddr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t baseAddr = reinterpret_cast<uintptr_t>(base);
  if (ptrAddr < baseAddr) {
    return false;
  }
  uintptr_t delta = ptrAddr - baseAddr;
  if (delta > bytes) {
    return false;
  }
  *offsetBytes = static_cast<int64_t>(delta);
  return true;
}

static flagcxResult_t
captureBufferRef(const void *ptr, const uniRunnerDagRuntimeBindings &bindings,
                 uniRunnerDagBufferRef *ref) {
  if (ptr == NULL) {
    ref->bufferType = uniRunnerDagBufferTypeNone;
    ref->offsetBytes = 0;
    return flagcxSuccess;
  }

  int64_t offsetBytes = 0;
  if (matchBindingRange(ptr, bindings.inputBase, bindings.inputBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeInput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (matchBindingRange(ptr, bindings.outputBase, bindings.outputBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeOutput;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (matchBindingRange(ptr, bindings.scratchBase, bindings.scratchBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeScratch;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  if (matchBindingRange(ptr, bindings.flagBase, bindings.flagBytes,
                        &offsetBytes)) {
    ref->bufferType = uniRunnerDagBufferTypeFlag;
    ref->offsetBytes = offsetBytes;
    return flagcxSuccess;
  }
  return flagcxInvalidArgument;
}

static flagcxResult_t
resolveBufferRef(const uniRunnerDagBufferRef &ref,
                 const uniRunnerDagRuntimeBindings &bindings, void **ptr) {
  const char *base = NULL;
  size_t bytes = 0;
  switch (ref.bufferType) {
    case uniRunnerDagBufferTypeNone:
      *ptr = NULL;
      return flagcxSuccess;
    case uniRunnerDagBufferTypeInput:
      base = static_cast<const char *>(bindings.inputBase);
      bytes = bindings.inputBytes;
      break;
    case uniRunnerDagBufferTypeOutput:
      base = static_cast<const char *>(bindings.outputBase);
      bytes = bindings.outputBytes;
      break;
    case uniRunnerDagBufferTypeScratch:
      base = static_cast<const char *>(bindings.scratchBase);
      bytes = bindings.scratchBytes;
      break;
    case uniRunnerDagBufferTypeFlag:
      base = static_cast<const char *>(bindings.flagBase);
      bytes = bindings.flagBytes;
      break;
    default:
      return flagcxInvalidArgument;
  }

  if (base == NULL || ref.offsetBytes < 0 ||
      static_cast<size_t>(ref.offsetBytes) > bytes) {
    return flagcxInvalidArgument;
  }

  *ptr = const_cast<char *>(base) + ref.offsetBytes;
  return flagcxSuccess;
}

static size_t
resolveEffectiveUniRunnerRedSlices(const flagcxUniRunnerState *runnerState,
                                   size_t count, int nranks) {
  if (runnerState->uniRunnerNRedSlices != 0) {
    return runnerState->uniRunnerNRedSlices;
  }
  if (count == 0 || runnerState->uniRunnerRedSliceSize == 0 ||
      runnerState->uniRunnerNSlices == 0 || nranks <= 0) {
    return 1;
  }

  size_t divisor = static_cast<size_t>(nranks) * runnerState->uniRunnerNSlices *
                   runnerState->uniRunnerRedSliceSize;
  if (divisor == 0) {
    return 1;
  }
  return std::max<size_t>(1, (count + divisor - 1) / divisor);
}

static uniRunnerDagCacheKey makeUniRunnerDagCacheKey(
    uniRunnerDagAlgoType algoType, flagcxCommOp_t commOp, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, int root, int groupSize,
    flagcxUniRunnerState *runnerState, flagcxComm_t comm, const void *sendbuff,
    void *recvbuff, void *scratchbuff) {
  uniRunnerDagCacheKey key{};
  key.algoType = algoType;
  key.commOp = commOp;
  key.count = count;
  key.datatype = datatype;
  key.redOp = redOp;
  key.rank = comm->rank;
  key.nranks = comm->nranks;
  key.root = root;
  key.groupSize = groupSize;
  key.numSlices = runnerState->uniRunnerNSlices;
  key.numRedSlices = runnerState->uniRunnerNRedSlices;
  key.redSliceSize = runnerState->uniRunnerRedSliceSize;
  key.nthreads = runnerState->uniRunnerNThreads;
  key.inputOutputAliased = sendbuff == recvbuff;
  key.inputScratchAliased = sendbuff == scratchbuff;
  key.outputScratchAliased = recvbuff == scratchbuff;
  return key;
}

static bool
findUniRunnerDagTemplate(size_t hashValue, const uniRunnerDagCacheKey &key,
                         std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  return findUniRunnerDagTemplateLocked(hashValue, key, dagTemplate);
}

static flagcxResult_t loadUniRunnerDagFromCacheDir(size_t hashValue, int rank) {
  std::string cacheDir = getUniRunnerDagCacheDirFromEnv();
  if (cacheDir.empty()) {
    return flagcxSuccess;
  }
  return loadUniRunnerDagFileIntoCache(
      makeUniRunnerDagFilePath(cacheDir, hashValue, rank));
}

static flagcxResult_t cacheUniRunnerDagTemplate(
    const std::shared_ptr<uniRunnerDagTemplate> &dagTemplate) {
  size_t hashValue = getUniRunnerDagPatternHash(dagTemplate->key);
  std::string dagDir = getUniRunnerDagCacheDirFromEnv();
  std::string dagPath =
      makeUniRunnerDagFilePath(dagDir, hashValue, dagTemplate->key.rank);
  {
    std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
    FLAGCXCHECK(insertUniRunnerDagTemplateLocked(dagTemplate));
  }
  if (dagDir.empty()) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(ensureUniRunnerDagDirExists(dagDir));
  FLAGCXCHECK(uniRunnerSaveDagJsonFile(dagPath, *dagTemplate));
  std::lock_guard<std::mutex> lock(gUniRunnerDagCacheMutex);
  gUniRunnerDagLoadedPaths.insert(dagPath);
  return flagcxSuccess;
}

} // namespace

size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key) {
  size_t hashValue = 0;
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.algoType));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.commOp));
  hashValue = hashCombine(hashValue, key.count);
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.datatype));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redOp));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.rank));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nranks));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.root + 1));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.groupSize + 1));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.numSlices));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.numRedSlices));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redSliceSize));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nthreads));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.inputOutputAliased));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.inputScratchAliased));
  hashValue =
      hashCombine(hashValue, static_cast<size_t>(key.outputScratchAliased));
  return hashValue;
}

static flagcxResult_t allocDagNodeDeps(uniRunnerDagNode *node) {
  node->pendingParents = 0;
  if (node->numParents > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->parents, node->numParents * sizeof(int)));
  }
  if (node->numChildren > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->children, node->numChildren * sizeof(int)));
  }
  return flagcxSuccess;
}

static flagcxResult_t setDagNodeParent(uniRunnerDagNode *node, int parentSlot,
                                       int parentIdx) {
  if (parentSlot < 0 || parentSlot >= node->numParents ||
      node->parents == NULL) {
    return flagcxInternalError;
  }
  node->parents[parentSlot] = parentIdx;
  node->pendingParents++;
  return flagcxSuccess;
}

// Validate that DAG construction filled every declared parent slot.
static flagcxResult_t validateDagNodes(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL || runnerState->dagNodes == NULL ||
      runnerState->numDagNodes == 0) {
    return flagcxSuccess;
  }

  const int numDagNodes = runnerState->numDagNodes;
  uniRunnerDagNode *dagNodes = runnerState->dagNodes;
  size_t numEdges = 0;

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    if (node->pendingParents != node->numParents) {
      return flagcxInternalError;
    }
    if (node->numParents < 0 || node->numChildren < 0) {
      return flagcxInternalError;
    }
    if ((node->numParents > 0 && node->parents == NULL) ||
        (node->numChildren > 0 && node->children == NULL)) {
      return flagcxInternalError;
    }
    numEdges += static_cast<size_t>(node->numParents);
  }

  std::unordered_set<uint64_t> dagEdges;
  dagEdges.reserve(numEdges);

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    for (int p = 0; p < node->numParents; p++) {
      int parentIdx = node->parents[p];
      if (parentIdx < 0 || parentIdx >= numDagNodes || parentIdx == i) {
        return flagcxInternalError;
      }
      uint64_t edge =
          (static_cast<uint64_t>(static_cast<uint32_t>(parentIdx)) << 32) |
          static_cast<uint32_t>(i);
      if (!dagEdges.emplace(edge).second) {
        return flagcxInternalError;
      }
    }
  }

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    for (int c = 0; c < node->numChildren; c++) {
      int childIdx = node->children[c];
      if (childIdx < 0 || childIdx >= numDagNodes || childIdx == i) {
        return flagcxInternalError;
      }
      uint64_t edge = (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
                      static_cast<uint32_t>(childIdx);
      std::unordered_set<uint64_t>::iterator it = dagEdges.find(edge);
      if (it == dagEdges.end()) {
        return flagcxInternalError;
      }
      dagEdges.erase(it);
    }
  }

  return dagEdges.empty() ? flagcxSuccess : flagcxInternalError;
}

static inline void *getDagNodeFlag(flagcxUniRunnerState *runnerState,
                                   int nodeIdx) {
  assert(nodeIdx >= 0 &&
         static_cast<size_t>(nodeIdx) < runnerState->streamFlagsSize);
  return runnerState->streamFlags[nodeIdx];
}

static void refreshStreamFlagAddressQueue(flagcxUniRunnerState *runnerState) {
  if (runnerState->streamFlagsPool == NULL ||
      runnerState->streamFlags == NULL) {
    return;
  }

  char *base = static_cast<char *>(runnerState->streamFlagsPool);
  for (size_t i = 0; i < runnerState->streamFlagsCapacity; ++i) {
    runnerState->streamFlags[i] =
        static_cast<void *>(base + i * sizeof(uint64_t));
  }
}

static flagcxResult_t
resizeStreamFlagAddressQueue(flagcxUniRunnerState *runnerState,
                             size_t newCapacity) {
  if (newCapacity <= runnerState->streamFlagsCapacity) {
    return flagcxSuccess;
  }

  if (runnerState->streamFlagsCapacity == 0) {
    if (runnerState->streamFlags != NULL) {
      free(runnerState->streamFlags);
      runnerState->streamFlags = NULL;
    }
    FLAGCXCHECK(flagcxCalloc(&runnerState->streamFlags, newCapacity));
  } else {
    FLAGCXCHECK(flagcxRealloc(&runnerState->streamFlags,
                              runnerState->streamFlagsCapacity, newCapacity));
  }

  return flagcxSuccess;
}

static flagcxResult_t
ensureStreamFlagQueueCapacity(flagcxUniRunnerState *runnerState,
                              size_t requiredFlags) {
  if (requiredFlags > runnerState->streamFlagsCapacity) {
    size_t newCapacity = runnerState->streamFlagsCapacity == 0
                             ? 1
                             : runnerState->streamFlagsCapacity;
    while (newCapacity < requiredFlags) {
      newCapacity *= 2;
    }

    FLAGCXCHECK(resizeStreamFlagAddressQueue(runnerState, newCapacity));

    void *newPool = NULL;
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(
        &newPool, newCapacity * sizeof(uint64_t), flagcxMemDevice, NULL));
    if (runnerState->streamFlagsPool != NULL) {
      FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                            flagcxMemDevice, NULL));
    }

    runnerState->streamFlagsPool = newPool;
    runnerState->streamFlagsCapacity = newCapacity;
    refreshStreamFlagAddressQueue(runnerState);
  }

  return flagcxSuccess;
}

static flagcxResult_t prepareDagStreamFlags(flagcxUniRunnerState *runnerState) {
  size_t activeFlags = runnerState->numDagNodes > 0
                           ? static_cast<size_t>(runnerState->numDagNodes)
                           : 0;
  FLAGCXCHECK(ensureStreamFlagQueueCapacity(runnerState, activeFlags));
  runnerState->streamFlagsSize = activeFlags;

  if (runnerState->streamFlagsSize != 0) {
    FLAGCXCHECK(deviceAdaptor->deviceMemset(runnerState->streamFlagsPool, 0,
                                            runnerState->streamFlagsSize *
                                                sizeof(uint64_t),
                                            flagcxMemDevice, NULL));
  }

  return flagcxSuccess;
}

static flagcxResult_t
destroyStreamFlagQueue(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL) {
    return flagcxSuccess;
  }

  if (runnerState->streamFlagsPool != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                          flagcxMemDevice, NULL));
    runnerState->streamFlagsPool = NULL;
  }

  if (runnerState->streamFlags != NULL) {
    free(runnerState->streamFlags);
    runnerState->streamFlags = NULL;
  }
  runnerState->streamFlagsSize = 0;
  runnerState->streamFlagsCapacity = 0;
  return flagcxSuccess;
}

static int getLocalRankForGlobalRank(flagcxComm_t comm, int globalRank) {
  if (comm == NULL || comm->localRankToRank == NULL) {
    return -1;
  }
  for (int lr = 0; lr < comm->localRanks; lr++) {
    if (comm->localRankToRank[lr] == globalRank) {
      return lr;
    }
  }
  return -1;
}

static void refreshDevApiFlagAddressQueue(flagcxUniRunnerState *runnerState) {
  if (runnerState->devApiFlagPool == NULL || runnerState->devApiFlags == NULL) {
    return;
  }

  char *base = static_cast<char *>(runnerState->devApiFlagPool);
  for (size_t i = 0; i < runnerState->devApiFlagsCapacity; i++) {
    runnerState->devApiFlags[i] =
        static_cast<void *>(base + i * sizeof(uint64_t));
  }
}

static flagcxResult_t resizeDevApiFlagAddressQueue(
    flagcxUniRunnerState *runnerState, size_t newCapacity) {
  if (newCapacity <= runnerState->devApiFlagsCapacity) {
    return flagcxSuccess;
  }

  if (runnerState->devApiFlagsCapacity == 0) {
    if (runnerState->devApiFlags != NULL) {
      free(runnerState->devApiFlags);
      runnerState->devApiFlags = NULL;
    }
    FLAGCXCHECK(flagcxCalloc(&runnerState->devApiFlags, newCapacity));
  } else {
    FLAGCXCHECK(flagcxRealloc(&runnerState->devApiFlags,
                              runnerState->devApiFlagsCapacity, newCapacity));
  }
  return flagcxSuccess;
}

static flagcxResult_t
destroyUniRunnerDevApiDataViews(flagcxUniRunnerState *runnerState) {
  flagcxComm_t comm = runnerState->devApiOwner;
  if (comm != NULL) {
    if (runnerState->devApiSendMem != NULL) {
      FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->devApiSendMem));
    }
    if (runnerState->devApiRecvMem != NULL) {
      FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->devApiRecvMem));
    }
  }
  runnerState->devApiSendMem = NULL;
  runnerState->devApiRecvMem = NULL;
  runnerState->devApiSendBase = NULL;
  runnerState->devApiRecvBase = NULL;
  runnerState->devApiDataBytes = 0;
  return flagcxSuccess;
}

static flagcxResult_t
destroyUniRunnerDevApiState(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL) {
    return flagcxSuccess;
  }

  flagcxComm_t comm = runnerState->devApiOwner;
  if (comm != NULL && runnerState->devApiComm != NULL) {
    FLAGCXCHECK(flagcxDevCommDestroy(comm, runnerState->devApiComm));
  }
  runnerState->devApiComm = NULL;

  FLAGCXCHECK(destroyUniRunnerDevApiDataViews(runnerState));

  if (comm != NULL && runnerState->devApiFlagMem != NULL) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->devApiFlagMem));
  }
  runnerState->devApiFlagMem = NULL;

  if (runnerState->devApiFlagPool != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->devApiFlagPool,
                                          flagcxMemDevice, NULL));
    runnerState->devApiFlagPool = NULL;
  }
  if (runnerState->devApiFlags != NULL) {
    free(runnerState->devApiFlags);
    runnerState->devApiFlags = NULL;
  }
  runnerState->devApiFlagsSize = 0;
  runnerState->devApiFlagsCapacity = 0;
  runnerState->devApiOwner = NULL;
  return flagcxSuccess;
}

static flagcxResult_t
ensureUniRunnerDevApiComm(flagcxUniRunnerState *runnerState, flagcxComm_t comm) {
  if (runnerState->devApiComm != NULL) {
    return flagcxSuccess;
  }
  if (comm == NULL || comm->localRanks <= 0) {
    return flagcxInvalidArgument;
  }

  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t previousDevCommHandle =
      comm->heteroComm == NULL ? NULL : comm->heteroComm->devCommHandle;
  runnerState->devApiOwner = comm;
  FLAGCXCHECK(flagcxDevCommCreate(comm, &reqs, &runnerState->devApiComm));
  if (comm->heteroComm != NULL &&
      previousDevCommHandle != runnerState->devApiComm) {
    comm->heteroComm->devCommHandle = previousDevCommHandle;
  }
  return flagcxSuccess;
}

static flagcxResult_t ensureUniRunnerDevApiDataViews(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm, const void *sendbuff,
    void *recvbuff, size_t bytes) {
  FLAGCXCHECK(ensureUniRunnerDevApiComm(runnerState, comm));

  if (runnerState->devApiSendMem != NULL &&
      runnerState->devApiRecvMem != NULL &&
      runnerState->devApiSendBase == sendbuff &&
      runnerState->devApiRecvBase == recvbuff &&
      runnerState->devApiDataBytes >= bytes) {
    return flagcxSuccess;
  }

  FLAGCXCHECK(destroyUniRunnerDevApiDataViews(runnerState));
  FLAGCXCHECK(flagcxDevMemCreate(comm, const_cast<void *>(sendbuff), bytes,
                                 NULL, &runnerState->devApiSendMem));
  FLAGCXCHECK(flagcxDevMemCreate(comm, recvbuff, bytes, NULL,
                                 &runnerState->devApiRecvMem));
  runnerState->devApiSendBase = sendbuff;
  runnerState->devApiRecvBase = recvbuff;
  runnerState->devApiDataBytes = bytes;
  return flagcxSuccess;
}

static flagcxResult_t ensureUniRunnerDevApiFlags(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    size_t requiredFlags) {
  FLAGCXCHECK(ensureUniRunnerDevApiComm(runnerState, comm));
  if (requiredFlags == 0) {
    runnerState->devApiFlagsSize = 0;
    return flagcxSuccess;
  }

  if (requiredFlags > runnerState->devApiFlagsCapacity) {
    size_t newCapacity = runnerState->devApiFlagsCapacity == 0
                             ? 1
                             : runnerState->devApiFlagsCapacity;
    while (newCapacity < requiredFlags) {
      newCapacity *= 2;
    }

    if (runnerState->devApiFlagMem != NULL) {
      FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->devApiFlagMem));
      runnerState->devApiFlagMem = NULL;
    }
    if (runnerState->devApiFlagPool != NULL) {
      FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->devApiFlagPool,
                                            flagcxMemDevice, NULL));
      runnerState->devApiFlagPool = NULL;
    }

    FLAGCXCHECK(resizeDevApiFlagAddressQueue(runnerState, newCapacity));
    FLAGCXCHECK(deviceAdaptor->deviceMalloc(
        &runnerState->devApiFlagPool, newCapacity * sizeof(uint64_t),
        flagcxMemDevice, NULL));
    runnerState->devApiFlagsCapacity = newCapacity;
    refreshDevApiFlagAddressQueue(runnerState);
    FLAGCXCHECK(flagcxDevMemCreate(comm, runnerState->devApiFlagPool,
                                   newCapacity * sizeof(uint64_t), NULL,
                                   &runnerState->devApiFlagMem));
  }

  runnerState->devApiFlagsSize = requiredFlags;
  FLAGCXCHECK(deviceAdaptor->deviceMemset(runnerState->devApiFlagPool, 0,
                                          requiredFlags * sizeof(uint64_t),
                                          flagcxMemDevice, NULL));
  // Remote signals must not race with another rank's local flag reset.
  FLAGCXCHECK(deviceAdaptor->deviceSynchronize());
  FLAGCXCHECK(bootstrapCollBarrier(comm->bootstrap, comm->rank, comm->nranks,
                                   0xDA71));
  return flagcxSuccess;
}

static void *getUniRunnerDevApiPeerPointer(flagcxComm_t comm,
                                           flagcxDevMem_t devMem,
                                           int globalRank,
                                           size_t offsetBytes) {
  if (comm == NULL || devMem == NULL) {
    return NULL;
  }

  int localRank = getLocalRankForGlobalRank(comm, globalRank);
  if (localRank < 0) {
    return NULL;
  }
  if (localRank == comm->localRank) {
    return static_cast<void *>(static_cast<char *>(devMem->rawPtr) +
                               offsetBytes);
  }
  if (devMem->ipcIndex < 0 || devMem->ipcIndex >= FLAGCX_MAX_IPC_ENTRIES) {
    return NULL;
  }
  flagcxIpcTableEntry *entry = &comm->ipcTable[devMem->ipcIndex];
  if (entry->hostPeerPtrs == NULL || localRank >= entry->nPeers ||
      entry->hostPeerPtrs[localRank] == NULL) {
    return NULL;
  }
  return static_cast<void *>(static_cast<char *>(entry->hostPeerPtrs[localRank]) +
                             offsetBytes);
}

static inline void *getUniRunnerDevApiLocalFlag(
    flagcxUniRunnerState *runnerState, size_t flagIdx) {
  assert(flagIdx < runnerState->devApiFlagsSize);
  return runnerState->devApiFlags[flagIdx];
}

static inline void *getUniRunnerDevApiPeerFlag(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm, int globalRank,
    size_t flagIdx) {
  assert(flagIdx < runnerState->devApiFlagsSize);
  return getUniRunnerDevApiPeerPointer(comm, runnerState->devApiFlagMem,
                                       globalRank,
                                       flagIdx * sizeof(uint64_t));
}

static inline flagcxStream_t
getDagNodeExecutionStream(flagcxUniRunnerState *runnerState,
                          const uniRunnerDagNode *node) {
  switch (node->nodeType) {
    case uniRunnerDagNodeTypeP2p:
      return runnerState->commStream;
    case uniRunnerDagNodeTypeRed:
    case uniRunnerDagNodeTypeDevApi:
      return runnerState->redStream;
    case uniRunnerDagNodeTypeCpy:
      return runnerState->cpyStream;
    default:
      return NULL;
  }
}

flagcxResult_t initUniRunnerStateDummy(flagcxUniRunnerState *runnerState) {
  return flagcxNotSupported;
}

static flagcxResult_t
buildUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 2) {
    return flagcxSuccess;
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateLocRed called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;
  size_t rankChunkCount =
      baseRankChunkCount + (rank < (int)rankChunkRemainder ? 1 : 0);

  const int numNodes = numSlices;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int s = 0; s < numSlices; s++) {
    size_t baseSliceCount = rankChunkCount / numSlices;
    size_t sliceRemainder = rankChunkCount % numSlices;
    // Calculate slice count with uneven distribution
    size_t sliceCount = baseSliceCount;
    if (s < sliceRemainder) {
      sliceCount++;
    }
    size_t sliceOffsetInChunk = s * baseSliceCount * typeSize;
    // Add offset for all previous slices that got the remainder
    sliceOffsetInChunk += std::min(s, (int)sliceRemainder) * typeSize;
    // Calculate offset accounting for rankChunkRemainder
    // First rankChunkRemainder ranks each have one extra element
    size_t rxOffset =
        (rank * baseRankChunkCount + std::min(rank, (int)rankChunkRemainder)) *
            typeSize +
        sliceOffsetInChunk;

    // Reduce Node
    int redNodeIdx = s;
    runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
    runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
    runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
    runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.input2 = static_cast<void *>(
        static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.output =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
    runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
        runnerState->uniRunnerNThreads;
    runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

    // Setup dependencies linearly within the slice chain
    runnerState->dagNodes[redNodeIdx].numParents = 0;
    runnerState->dagNodes[redNodeIdx].numChildren = 0;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
    // Enqueue the head of this slice chain to Ready Queue
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[redNodeIdx]);
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateGroupedAG(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxComm_t comm, int groupSize) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (groupSize <= 0 || groupSize > nranks || nranks % groupSize != 0) {
    return flagcxInvalidArgument;
  }

  int nGroups = nranks / groupSize;
  int groupIdx = rank / groupSize;
  int locRank = rank % groupSize;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateGroupedAG called, count=%lu, groupSize=%d, "
        "nGroups=%d",
        rank, count, groupSize, nGroups);

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t groupChunkCount = count * groupSize;
  const int numNodes = nGroups + 1;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  size_t localBaseOffset =
      static_cast<size_t>(groupIdx) * groupChunkCount * typeSize;
  for (int step = 0; step < nGroups; step++) {
    int nodeIdx = step;
    bool isLastStep = (step == nGroups - 1);
    int numIntraPeers = groupSize - 1;
    int numOps = isLastStep ? 2 * numIntraPeers : 2 * numIntraPeers + 2;

    runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
    runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
    runnerState->dagNodes[nodeIdx].nodeData.p2p.numOps = numOps;
    if (numOps > 0) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[nodeIdx].nodeData.p2p.ops,
                               numOps * sizeof(struct uniRunnerP2pOpData)));
    }

    for (int i = 0; i < numIntraPeers; i++) {
      int locSendPeer = (locRank + i + 1) % groupSize;
      int locRecvPeer = (locRank - i - 1 + groupSize) % groupSize;

      // Send
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].peerRank =
          groupIdx * groupSize + locSendPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i].addr =
          step == 0 ? const_cast<void *>(sendbuff)
                    : static_cast<void *>(static_cast<char *>(recvbuff) +
                                          localBaseOffset +
                                          locRank * count * typeSize);
      // Recv
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].peerRank =
          groupIdx * groupSize + locRecvPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[2 * i + 1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + localBaseOffset +
                              locRecvPeer * count * typeSize);
      TRACE(FLAGCX_UNIRUNNER,
            "Node %d: intra-group step %d, sendPeer=%d, recvPeer=%d, "
            "sendOffset=%lu, recvOffset=%lu",
            nodeIdx, i, groupIdx * groupSize + locSendPeer,
            groupIdx * groupSize + locRecvPeer,
            localBaseOffset + locRank * count * typeSize,
            localBaseOffset + locRecvPeer * count * typeSize);
    }

    if (!isLastStep) {
      size_t sendGroupIdx = (groupIdx + step + 1) % nGroups;
      size_t recvGroupIdx = (groupIdx - step - 1 + nGroups) % nGroups;
      size_t sendPeer = sendGroupIdx * groupSize + locRank;
      size_t recvPeer = recvGroupIdx * groupSize + locRank;
      size_t recvOffset = recvPeer * count * typeSize;
      int opIdx = 2 * numIntraPeers;

      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].peerRank =
          sendPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].addr =
          step == 0 ? const_cast<void *>(sendbuff)
                    : static_cast<void *>(static_cast<char *>(recvbuff) +
                                          localBaseOffset +
                                          locRank * count * typeSize);
      opIdx++;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].peerRank =
          recvPeer;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].count = count;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].datatype =
          datatype;
      runnerState->dagNodes[nodeIdx].nodeData.p2p.ops[opIdx].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + recvOffset);

      TRACE(FLAGCX_UNIRUNNER,
            "Node %d: inter-group step %d, sendPeer=%lu, recvPeer=%lu, "
            "recvOffset=%lu",
            nodeIdx, step, sendPeer, recvPeer, recvOffset);

      localBaseOffset = recvGroupIdx * groupChunkCount * typeSize;
    }
  }

  int nodeIdx = numNodes - 1;
  runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
  runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeCpy;
  runnerState->dagNodes[nodeIdx].nodeData.cpy.src =
      const_cast<void *>(sendbuff);
  runnerState->dagNodes[nodeIdx].nodeData.cpy.dst = static_cast<void *>(
      static_cast<char *>(recvbuff) + rank * count * typeSize);
  runnerState->dagNodes[nodeIdx].nodeData.cpy.count = count;
  runnerState->dagNodes[nodeIdx].nodeData.cpy.datatype = datatype;

  for (int s = 0; s < nGroups; s++) {
    runnerState->dagNodes[s].numParents = (s == 0) ? 0 : 1;
    runnerState->dagNodes[s].numChildren = (s == nGroups - 1) ? 0 : 1;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[s]));

    if (s == 0) {
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[s]);
    } else {
      runnerState->numPendingNodes++;
      FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[s], 0, s - 1));
    }
    if (s != nGroups - 1) {
      runnerState->dagNodes[s].children[0] = s + 1;
    }
  }

  runnerState->dagNodes[nodeIdx].numParents = 0;
  runnerState->dagNodes[nodeIdx].numChildren = 0;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[nodeIdx]));
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[nodeIdx]);

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Grouped AllGather topology",
        nranks);
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d",
          i,
          (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
          : (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeRed)
              ? "RED"
              : "CPY",
          runnerState->dagNodes[i].numParents,
          runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateP2p called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes + 1;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = i
   */
  for (int s = 0; s < numSlices; s++) {
    // All-Gather
    int sliceNodeBaseIdx = globalNodeIdx;
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, baseIdx %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, sliceNodeBaseIdx, txRankChunkCount, txSliceCount,
            rxRankChunkCount, rxSliceCount, txChunk, txOffset, rxChunk,
            rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    }

    // Setup dependencies linearly within the slice chain
    for (int i = 0; i < nodesPerSlice; i++) {
      int currIdx = sliceNodeBaseIdx + i;

      if (currIdx == 0) {
        runnerState->dagNodes[currIdx].numParents = 0;
      } else {
        runnerState->dagNodes[currIdx].numParents = 1;
      }
      if (currIdx == numNodes - 1) {
        runnerState->dagNodes[currIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[currIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[currIdx]));
      if (currIdx != 0) {
        int parentIdx = s == 0 ? (numSlices - 1) * nodesPerSlice + i - 1
                               : currIdx - nodesPerSlice;
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[currIdx], 0, parentIdx));
      }
      if (s == numSlices - 1) {
        if (currIdx != numNodes - 1) {
          runnerState->dagNodes[currIdx].children[0] = i + 1;
        }
      } else {
        runnerState->dagNodes[currIdx].children[0] = currIdx + nodesPerSlice;
      }
    }
  }
  // Copy local chunk from sendbuff to recvbuff before starting AG
  // Calculate offset accounting for rankChunkRemainder
  // First rankChunkRemainder ranks each have one extra element
  size_t localRankChunkCount =
      baseRankChunkCount + (rank < (int)rankChunkRemainder ? 1 : 0);
  size_t localChunkOffset =
      (rank * baseRankChunkCount + std::min(rank, (int)rankChunkRemainder)) *
      typeSize;
  int cpyNodeIdx = globalNodeIdx++;
  runnerState->dagNodes[cpyNodeIdx].nodeIdx = cpyNodeIdx;
  runnerState->dagNodes[cpyNodeIdx].nodeType = uniRunnerDagNodeTypeCpy;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.src = static_cast<void *>(
      static_cast<char *>(const_cast<void *>(sendbuff)) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.dst =
      static_cast<void *>(static_cast<char *>(recvbuff) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.count = localRankChunkCount;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.datatype = datatype;
  runnerState->dagNodes[cpyNodeIdx].numParents = 0;
  runnerState->dagNodes[cpyNodeIdx].numChildren = 0;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[cpyNodeIdx]));
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[cpyNodeIdx]);
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[0]);
  runnerState->numPendingNodes = numNodes - 1;

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                          const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int numSlices = runnerState->uniRunnerNSlices;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingAR called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = 3 * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * 2
   * redNodeIdx = s * nodesPerSlice + i * 2 + 1
   * all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + (nranks - 1) * 2 + i
   */
  for (int s = 0; s < numSlices; s++) {
    // Phase 1: Scatter-Reduce
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 2;
        }
        runnerState->numPendingNodes++;
      }
      runnerState->dagNodes[p2pNodeIdx].numChildren = 2;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice + 2 * (i - 1);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 1,
                                       p2pNodeIdx - 1));
        }
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * (i + 1);
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, 2 * (i + 1));
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, p2pNodeIdx + nodesPerSlice);
      }
      runnerState->dagNodes[p2pNodeIdx].children[1] = p2pNodeIdx + 1;
      TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
            p2pNodeIdx, p2pNodeIdx + 1);

      // Reduce Node
      int redNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
      runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
      runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
      runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
          static_cast<void *>(
              static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.output =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.count = rxSliceCount;
      runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
          runnerState->uniRunnerNThreads;
      runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

      // Set up red node dependency
      runnerState->numPendingNodes++;
      runnerState->dagNodes[redNodeIdx].numParents = 1;
      runnerState->dagNodes[redNodeIdx].numChildren = 1;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0, p2pNodeIdx));
      runnerState->dagNodes[redNodeIdx].children[0] = redNodeIdx + 1;
      TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
            redNodeIdx, redNodeIdx + 1);
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up all-gather phase p2p node dependency
      runnerState->numPendingNodes++;
      if (i == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 2;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1;
      }
      if (p2pNodeIdx == numNodes - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      int parentIdx = p2pNodeIdx - nodesPerSlice;
      if (s == 0) {
        if (i == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice + 2 * (nranks - 2);
        } else {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + 2 * (nranks - 1) + i - 1;
        }
      }
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
      if (i == 0) {
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 1,
                                     p2pNodeIdx - 1));
      }
      if (s == numSlices - 1) {
        if (p2pNodeIdx != numNodes - 1) {
          runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * nranks + i - 1;
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
                p2pNodeIdx, 2 * nranks + i - 1);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
              p2pNodeIdx, p2pNodeIdx + nodesPerSlice);
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Ring AllReduce topology (%d "
        "slices)",
        nranks, numSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t
buildUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                            const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateSlicedAR called, count=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce * numRedSlices) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = (numRedSlices + 2) * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1
   * all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + (nranks - 1) * (1 + numRedSlices) + i
   */
  for (int s = 0; s < numSlices; s++) {
    // Phase 1: Scatter-Reduce
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      runnerState->dagNodes[p2pNodeIdx].numChildren = 1 + numRedSlices;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                         r + 1, p2pNodeIdx - numRedSlices + r));
          }
        }
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            (i + 1) * (1 + numRedSlices);
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 0: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[p2pNodeIdx].children[r + 1] = p2pNodeIdx + 1 + r;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, r + 1,
              runnerState->dagNodes[p2pNodeIdx].children[r + 1]);
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = rxSliceCount / numRedSlices;
      size_t redSliceRemainder = rxSliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(recvbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(
                static_cast<char *>(const_cast<void *>(sendbuff)) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            static_cast<void *>(static_cast<char *>(recvbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        runnerState->dagNodes[redNodeIdx].numChildren = 1;
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     p2pNodeIdx));
        runnerState->dagNodes[redNodeIdx].children[0] =
            redSliceStartIdx + numRedSlices;
        TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
              redNodeIdx, runnerState->dagNodes[redNodeIdx].children[0]);
      }
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;

      // Calculate slice count with uneven distribution (last slice gets
      // remainder)
      size_t txRankChunkCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankChunkCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      // Calculate offsets accounting for rankChunkRemainder
      // First rankChunkRemainder ranks each have one extra element
      size_t txOffset = (txChunk * baseRankChunkCount +
                         std::min(txChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        txSliceOffsetInChunk;
      size_t rxOffset = (rxChunk * baseRankChunkCount +
                         std::min(rxChunk, (int)rankChunkRemainder)) *
                            typeSize +
                        rxSliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up all-gather phase p2p node dependency
      runnerState->numPendingNodes++;
      if (i == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1;
      }
      if (p2pNodeIdx == numNodes - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      int parentIdx = p2pNodeIdx - nodesPerSlice;
      if (s == 0) {
        if (i == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (nranks - 2) * (1 + numRedSlices);
        } else {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (nranks - 1) * (1 + numRedSlices) + i - 1;
        }
      }
      FLAGCXCHECK(
          setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
      if (i == 0) {
        for (int r = 0; r < numRedSlices; r++) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                       r + 1, p2pNodeIdx - numRedSlices + r));
        }
      }
      if (s == numSlices - 1) {
        if (p2pNodeIdx != numNodes - 1) {
          runnerState->dagNodes[p2pNodeIdx].children[0] =
              (1 + numRedSlices) * (nranks - 1) + i + 1;
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
                p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child 1: %d", rank,
              p2pNodeIdx, runnerState->dagNodes[p2pNodeIdx].children[0]);
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Sliced AllReduce topology (%d "
        "slices, %d redSlices)",
        nranks, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static void initDevApiDagNode(uniRunnerDagNode *node, int nodeIdx,
                              flagcxDevicePrim prim, void *sendbuff,
                              void *recvbuff, void *flag, size_t count,
                              size_t nthreads, flagcxDataType_t datatype,
                              int peerRank) {
  node->nodeIdx = nodeIdx;
  node->nodeType = uniRunnerDagNodeTypeDevApi;
  node->nodeData.devapi.sendbuff = sendbuff;
  node->nodeData.devapi.recvbuff = recvbuff;
  node->nodeData.devapi.flag = flag;
  node->nodeData.devapi.count = count;
  node->nodeData.devapi.nthreads = nthreads;
  node->nodeData.devapi.datatype = datatype;
  node->nodeData.devapi.type = prim;
  node->nodeData.devapi.peerRank = peerRank;
  node->nodeData.devapi.triggerIdx = -1;
}

static void initCpyDagNode(uniRunnerDagNode *node, int nodeIdx, void *src,
                           void *dst, void *waitFlag, size_t count,
                           flagcxDataType_t datatype) {
  node->nodeIdx = nodeIdx;
  node->nodeType = uniRunnerDagNodeTypeCpy;
  node->nodeData.cpy.src = src;
  node->nodeData.cpy.dst = dst;
  node->nodeData.cpy.waitFlag = waitFlag;
  node->nodeData.cpy.count = count;
  node->nodeData.cpy.datatype = datatype;
}

static void freeDagNodeStorage(uniRunnerDagNode *dagNodes, int numDagNodes) {
  if (dagNodes == NULL) {
    return;
  }
  for (int i = 0; i < numDagNodes; i++) {
    if (dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
        dagNodes[i].nodeData.p2p.ops != NULL) {
      free(dagNodes[i].nodeData.p2p.ops);
    }
    if (dagNodes[i].parents != NULL) {
      free(dagNodes[i].parents);
    }
    if (dagNodes[i].children != NULL) {
      free(dagNodes[i].children);
    }
  }
  free(dagNodes);
}

static bool getBufferOffsetBytes(const void *ptr, const void *base,
                                 size_t bytes, size_t *offsetBytes) {
  if (ptr == NULL || base == NULL) {
    return false;
  }
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t b = reinterpret_cast<uintptr_t>(base);
  if (p < b || p - b >= bytes) {
    return false;
  }
  *offsetBytes = static_cast<size_t>(p - b);
  return true;
}

static flagcxResult_t getDevApiPointerForLocalBuffer(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm, const void *sendbuff,
    void *recvbuff, size_t totalBytes, const void *ptr, int globalRank,
    void **devPtr, size_t *offsetBytes) {
  flagcxDevMem_t devMem = NULL;
  size_t offset = 0;

  // Prefer recvbuff when buffers alias; either DevMem points at the same raw
  // address, and recvbuff is the destination surface for the ring pipeline.
  if (getBufferOffsetBytes(ptr, recvbuff, totalBytes, &offset)) {
    devMem = runnerState->devApiRecvMem;
  } else if (getBufferOffsetBytes(ptr, sendbuff, totalBytes, &offset)) {
    devMem = runnerState->devApiSendMem;
  } else {
    return flagcxInvalidArgument;
  }

  void *peerPtr =
      getUniRunnerDevApiPeerPointer(comm, devMem, globalRank, offset);
  if (peerPtr == NULL) {
    return flagcxNotSupported;
  }
  *devPtr = peerPtr;
  if (offsetBytes != NULL) {
    *offsetBytes = offset;
  }
  return flagcxSuccess;
}

static const uniRunnerP2pOpData *
findP2pOpOfType(const uniRunnerDagNode *node, flagcxDevicePrim type) {
  if (node == NULL || node->nodeType != uniRunnerDagNodeTypeP2p) {
    return NULL;
  }
  for (int i = 0; i < node->nodeData.p2p.numOps; i++) {
    const uniRunnerP2pOpData *op = &node->nodeData.p2p.ops[i];
    if (op->type == type) {
      return op;
    }
  }
  return NULL;
}

static flagcxResult_t addDevARDagEdge(std::vector<std::vector<int>> *parents,
                                      std::vector<std::vector<int>> *children,
                                      int parentIdx, int childIdx) {
  if (parentIdx < 0 || childIdx < 0 ||
      parentIdx >= static_cast<int>(children->size()) ||
      childIdx >= static_cast<int>(parents->size()) ||
      parentIdx == childIdx) {
    return flagcxInternalError;
  }
  (*children)[parentIdx].push_back(childIdx);
  (*parents)[childIdx].push_back(parentIdx);
  return flagcxSuccess;
}

static flagcxResult_t finalizeDevARDag(flagcxUniRunnerState *runnerState,
                                       uniRunnerDagNode *newDag, int newNum,
                                       uniRunnerDagNode *oldDag, int oldNum,
                                       std::vector<std::vector<int>> &parents,
                                       std::vector<std::vector<int>> &children) {
  for (int i = 0; i < newNum; i++) {
    newDag[i].numParents = static_cast<int>(parents[i].size());
    newDag[i].numChildren = static_cast<int>(children[i].size());
    FLAGCXCHECK(allocDagNodeDeps(&newDag[i]));
    for (int p = 0; p < newDag[i].numParents; p++) {
      FLAGCXCHECK(setDagNodeParent(&newDag[i], p, parents[i][p]));
    }
    for (int c = 0; c < newDag[i].numChildren; c++) {
      newDag[i].children[c] = children[i][c];
    }
  }

  freeDagNodeStorage(oldDag, oldNum);
  runnerState->dagNodes = newDag;
  runnerState->numDagNodes = newNum;
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  runnerState->numPendingNodes = 0;
  for (int i = 0; i < newNum; i++) {
    if (newDag[i].numParents == 0) {
      if (newDag[i].nodeType == uniRunnerDagNodeTypeRed ||
          newDag[i].nodeType == uniRunnerDagNodeTypeDevApi) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, &newDag[i]);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, &newDag[i]);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }
  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateDevAR(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm, bool useGet) {
  if (comm->localRanks != comm->nranks) {
    return flagcxNotSupported;
  }
  if (count == 0 || comm->nranks <= 1) {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  }

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0) {
    return flagcxNotSupported;
  }
  size_t totalBytes = count * typeSize;

  FLAGCXCHECK(buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                          count, datatype, op, comm));
  uniRunnerDagNode *oldDag = runnerState->dagNodes;
  int oldNum = runnerState->numDagNodes;

  int p2pCount = 0;
  int newNum = 0;
  for (int i = 0; i < oldNum; i++) {
    if (oldDag[i].nodeType == uniRunnerDagNodeTypeP2p) {
      p2pCount++;
      newNum += (oldDag[i].numParents > 1 ? 1 : 0) + (useGet ? 3 : 2);
    } else {
      newNum++;
    }
  }

  FLAGCXCHECK(ensureUniRunnerDevApiDataViews(runnerState, comm, sendbuff,
                                             recvbuff, totalBytes));
  FLAGCXCHECK(ensureUniRunnerDevApiFlags(runnerState, comm, p2pCount));

  uniRunnerDagNode *newDag = NULL;
  FLAGCXCHECK(flagcxCalloc(&newDag, newNum * sizeof(struct uniRunnerDagNode)));
  if (newDag == NULL) {
    return flagcxSystemError;
  }

  std::vector<std::vector<int>> parents(newNum);
  std::vector<std::vector<int>> children(newNum);
  std::vector<int> startMap(oldNum, -1);
  std::vector<int> completeMap(oldNum, -1);

  int nextRank = (comm->rank + 1) % comm->nranks;
  int prevRank = (comm->rank - 1 + comm->nranks) % comm->nranks;
  int newIdx = 0;
  int p2pOrdinal = 0;

  for (int oldIdx = 0; oldIdx < oldNum; oldIdx++) {
    uniRunnerDagNode *oldNode = &oldDag[oldIdx];
    if (oldNode->nodeType == uniRunnerDagNodeTypeRed) {
      int redIdx = newIdx++;
      newDag[redIdx].nodeIdx = redIdx;
      newDag[redIdx].nodeType = uniRunnerDagNodeTypeRed;
      newDag[redIdx].nodeData.red = oldNode->nodeData.red;
      newDag[redIdx].nodeData.red.triggerIdx = -1;
      startMap[oldIdx] = redIdx;
      completeMap[oldIdx] = redIdx;
      continue;
    }
    if (oldNode->nodeType != uniRunnerDagNodeTypeP2p) {
      return flagcxNotSupported;
    }

    const uniRunnerP2pOpData *sendOp =
        findP2pOpOfType(oldNode, flagcxDevicePrimSend);
    const uniRunnerP2pOpData *recvOp =
        findP2pOpOfType(oldNode, flagcxDevicePrimRecv);
    if (sendOp == NULL || recvOp == NULL) {
      return flagcxNotSupported;
    }

    int joinIdx = -1;
    if (oldNode->numParents > 1) {
      joinIdx = newIdx++;
      initCpyDagNode(&newDag[joinIdx], joinIdx, NULL, NULL, NULL, 0, datatype);
    }

    size_t sendOffset = 0;
    void *localSend = NULL;
    FLAGCXCHECK(getDevApiPointerForLocalBuffer(
        runnerState, comm, sendbuff, recvbuff, totalBytes, sendOp->addr,
        comm->rank, &localSend, &sendOffset));

    void *remoteRecv =
        useGet ? NULL
               : getUniRunnerDevApiPeerPointer(
                     comm, runnerState->devApiRecvMem, nextRank, sendOffset);
    void *remoteFlag =
        getUniRunnerDevApiPeerFlag(runnerState, comm, nextRank, p2pOrdinal);
    void *localFlag = getUniRunnerDevApiLocalFlag(runnerState, p2pOrdinal);
    if ((!useGet && remoteRecv == NULL) || remoteFlag == NULL ||
        localFlag == NULL) {
      return flagcxNotSupported;
    }

    if (useGet) {
      int signalIdx = newIdx++;
      int waitReadyIdx = newIdx++;
      int getIdx = newIdx++;

      initDevApiDagNode(&newDag[signalIdx], signalIdx, flagcxDevicePrimPut,
                        NULL, NULL, remoteFlag, 0,
                        runnerState->uniRunnerNThreads, datatype, nextRank);
      initCpyDagNode(&newDag[waitReadyIdx], waitReadyIdx, NULL, NULL,
                     localFlag, 0, datatype);

      size_t recvOffset = 0;
      void *localRecv = NULL;
      FLAGCXCHECK(getDevApiPointerForLocalBuffer(
          runnerState, comm, sendbuff, recvbuff, totalBytes, recvOp->addr,
          comm->rank, &localRecv, &recvOffset));

      flagcxDevMem_t remoteSrcMem =
          getBufferOffsetBytes(sendOp->addr, recvbuff, totalBytes,
                               &sendOffset)
              ? runnerState->devApiRecvMem
              : runnerState->devApiSendMem;
      void *remoteSrc =
          getUniRunnerDevApiPeerPointer(comm, remoteSrcMem, prevRank,
                                        recvOffset);
      if (remoteSrc == NULL || localRecv == NULL) {
        return flagcxNotSupported;
      }

      initDevApiDagNode(&newDag[getIdx], getIdx, flagcxDevicePrimGet,
                        remoteSrc, localRecv, NULL, recvOp->count,
                        runnerState->uniRunnerNThreads, datatype, prevRank);

      if (joinIdx >= 0) {
        FLAGCXCHECK(addDevARDagEdge(&parents, &children, joinIdx, signalIdx));
      }
      FLAGCXCHECK(
          addDevARDagEdge(&parents, &children, signalIdx, waitReadyIdx));
      FLAGCXCHECK(addDevARDagEdge(&parents, &children, waitReadyIdx, getIdx));
      startMap[oldIdx] = joinIdx >= 0 ? joinIdx : signalIdx;
      completeMap[oldIdx] = getIdx;
    } else {
      int putIdx = newIdx++;
      int waitIdx = newIdx++;

      initDevApiDagNode(&newDag[putIdx], putIdx, flagcxDevicePrimPut,
                        localSend, remoteRecv, remoteFlag, sendOp->count,
                        runnerState->uniRunnerNThreads, datatype, nextRank);
      initCpyDagNode(&newDag[waitIdx], waitIdx, NULL, NULL, localFlag, 0,
                     datatype);

      if (joinIdx >= 0) {
        FLAGCXCHECK(addDevARDagEdge(&parents, &children, joinIdx, putIdx));
      }
      FLAGCXCHECK(addDevARDagEdge(&parents, &children, putIdx, waitIdx));
      startMap[oldIdx] = joinIdx >= 0 ? joinIdx : putIdx;
      completeMap[oldIdx] = waitIdx;
    }
    p2pOrdinal++;
  }

  if (newIdx != newNum || p2pOrdinal != p2pCount) {
    return flagcxInternalError;
  }

  for (int oldIdx = 0; oldIdx < oldNum; oldIdx++) {
    for (int c = 0; c < oldDag[oldIdx].numChildren; c++) {
      int oldChild = oldDag[oldIdx].children[c];
      FLAGCXCHECK(addDevARDagEdge(&parents, &children, completeMap[oldIdx],
                                  startMap[oldChild]));
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initialized DevAR %s from SlicedAR, oldNodes=%d newNodes=%d "
        "p2p=%d",
        comm->rank, useGet ? "get" : "put", oldNum, newNum, p2pCount);

  return finalizeDevARDag(runnerState, newDag, newNum, oldDag, oldNum, parents,
                          children);
}

static flagcxResult_t buildUniRunnerStateDevApiTest(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm) {
  (void)op;
  if (comm->nranks < 1) {
    return flagcxInvalidArgument;
  }
  if (count == 0) {
    return flagcxSuccess;
  }
  if (comm->localRanks != comm->nranks) {
    return flagcxNotSupported;
  }

  const int rank = comm->rank;
  const int nranks = comm->nranks;
  const int nextRank = (rank + 1) % nranks;
  const int prevRank = (rank - 1 + nranks) % nranks;
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0) {
    return flagcxNotSupported;
  }
  const size_t totalBytes = count * typeSize;
  const size_t putCount = (count + 1) / 2;
  const size_t getCount = count - putCount;
  const size_t getOffsetBytes = putCount * typeSize;
  const int numNodes = getCount == 0 ? 2 : 4;

  FLAGCXCHECK(ensureUniRunnerDevApiDataViews(runnerState, comm, sendbuff,
                                             recvbuff, totalBytes));
  FLAGCXCHECK(ensureUniRunnerDevApiFlags(runnerState, comm, 2));

  void *putSrc =
      getUniRunnerDevApiPeerPointer(comm, runnerState->devApiSendMem, rank, 0);
  void *putDst = getUniRunnerDevApiPeerPointer(
      comm, runnerState->devApiRecvMem, nextRank, 0);
  void *putFlag = getUniRunnerDevApiPeerFlag(runnerState, comm, nextRank, 0);
  void *waitPutFlag = getUniRunnerDevApiLocalFlag(runnerState, 0);
  if (putSrc == NULL || putDst == NULL || putFlag == NULL ||
      waitPutFlag == NULL) {
    return flagcxNotSupported;
  }

  void *getSrc = NULL;
  void *getDst = NULL;
  void *getFlag = NULL;
  void *waitGetFlag = NULL;
  if (getCount != 0) {
    getSrc = getUniRunnerDevApiPeerPointer(
        comm, runnerState->devApiSendMem, prevRank, getOffsetBytes);
    getDst = getUniRunnerDevApiPeerPointer(
        comm, runnerState->devApiRecvMem, rank, getOffsetBytes);
    getFlag = getUniRunnerDevApiLocalFlag(runnerState, 1);
    waitGetFlag = getUniRunnerDevApiLocalFlag(runnerState, 1);
    if (getSrc == NULL || getDst == NULL || getFlag == NULL ||
        waitGetFlag == NULL) {
      return flagcxNotSupported;
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateDevApiTest count=%lu putCount=%lu "
        "getCount=%lu prev=%d next=%d",
        rank, count, putCount, getCount, prevRank, nextRank);

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  initDevApiDagNode(&runnerState->dagNodes[0], 0, flagcxDevicePrimPut, putSrc,
                    putDst, putFlag, putCount, runnerState->uniRunnerNThreads,
                    datatype, nextRank);
  initDevApiDagNode(&runnerState->dagNodes[1], 1, flagcxDevicePrimWaitSignal,
                    NULL, NULL, waitPutFlag, 0,
                    runnerState->uniRunnerNThreads, datatype, prevRank);

  runnerState->dagNodes[0].numParents = 0;
  runnerState->dagNodes[0].numChildren = 1;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
  runnerState->dagNodes[0].children[0] = 1;
  flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                          &runnerState->dagNodes[0]);

  runnerState->dagNodes[1].numParents = 1;
  runnerState->dagNodes[1].numChildren = getCount == 0 ? 0 : 1;
  FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[1]));
  FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[1], 0, 0));
  if (getCount != 0) {
    runnerState->dagNodes[1].children[0] = 2;
  }
  runnerState->numPendingNodes++;

  if (getCount != 0) {
    initDevApiDagNode(&runnerState->dagNodes[2], 2, flagcxDevicePrimGet,
                      getSrc, getDst, getFlag, getCount,
                      runnerState->uniRunnerNThreads, datatype, prevRank);
    initDevApiDagNode(&runnerState->dagNodes[3], 3,
                      flagcxDevicePrimWaitSignal, NULL, NULL, waitGetFlag, 0,
                      runnerState->uniRunnerNThreads, datatype, rank);

    runnerState->dagNodes[2].numParents = 1;
    runnerState->dagNodes[2].numChildren = 1;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[2]));
    FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[2], 0, 1));
    runnerState->dagNodes[2].children[0] = 3;
    runnerState->numPendingNodes++;

    runnerState->dagNodes[3].numParents = 1;
    runnerState->dagNodes[3].numChildren = 0;
    FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[3]));
    FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[3], 0, 2));
    runnerState->numPendingNodes++;
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateRingRS(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    void *scratchbuff, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t op, flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingRS called, recvcount=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t baseRankChunkCount = count;

  // Nodes per slice chain:
  // (P2P + Reduce * numRedSlices) * (nranks - 1)
  const int nodesPerSlice = (numRedSlices + 1) * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* reduce-scatter (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1
   */
  for (int s = 0; s < numSlices; s++) {
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeIdx = p2pNodeIdx;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i - 1 + nranks) % nranks;
      int rxChunk = (rank - i - 2 + nranks) % nranks;

      size_t txRankChunkCount = baseRankChunkCount;
      size_t rxRankChunkCount = baseRankChunkCount;
      size_t txBaseSliceCount = txRankChunkCount / numSlices;
      size_t rxBaseSliceCount = rxRankChunkCount / numSlices;
      size_t txSliceRemainder = txRankChunkCount % numSlices;
      size_t rxSliceRemainder = rxRankChunkCount % numSlices;
      size_t txSliceCount = txBaseSliceCount + (s < txSliceRemainder ? 1 : 0);
      size_t rxSliceCount = rxBaseSliceCount + (s < rxSliceRemainder ? 1 : 0);
      size_t txSliceOffsetInChunk = s * txBaseSliceCount * typeSize;
      txSliceOffsetInChunk += std::min(s, (int)txSliceRemainder) * typeSize;
      size_t rxSliceOffsetInChunk = s * rxBaseSliceCount * typeSize;
      rxSliceOffsetInChunk += std::min(s, (int)rxSliceRemainder) * typeSize;

      size_t txOffset =
          (txChunk * baseRankChunkCount) * typeSize + txSliceOffsetInChunk;
      size_t rxOffset =
          (rxChunk * baseRankChunkCount) * typeSize + rxSliceOffsetInChunk;

      TRACE(FLAGCX_UNIRUNNER,
            "Initializing rank %d slice %d, step %d, txRankCount "
            "%lu, txSliceCount %lu, rxRankCount %lu, rxSliceCount %lu, tx "
            "chunk %d off %lu, rx chunk %d off %lu",
            rank, s, i, txRankChunkCount, txSliceCount, rxRankChunkCount,
            rxSliceCount, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count =
          txSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from scratchbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : scratchbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count =
          rxSliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(scratchbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      if (i == nranks - 2 && s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = numRedSlices;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1 + numRedSlices;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[p2pNodeIdx]));
      if (p2pNodeIdx != 0) {
        int parentIdx = p2pNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(
            setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx], 0, parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[p2pNodeIdx],
                                         r + 1, p2pNodeIdx - numRedSlices + r));
          }
        }
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[p2pNodeIdx].children[r] = p2pNodeIdx + 1 + r;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, r, runnerState->dagNodes[p2pNodeIdx].children[r]);
      }
      if (s == numSlices - 1) {
        if (i != nranks - 2) {
          runnerState->dagNodes[p2pNodeIdx].children[numRedSlices] =
              (i + 1) * (1 + numRedSlices);
          TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
                p2pNodeIdx, numRedSlices,
                runnerState->dagNodes[p2pNodeIdx].children[numRedSlices]);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[numRedSlices] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_UNIRUNNER, "rank %d p2pNode %d child %d: %d", rank,
              p2pNodeIdx, numRedSlices,
              runnerState->dagNodes[p2pNodeIdx].children[numRedSlices]);
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = rxSliceCount / numRedSlices;
      size_t redSliceRemainder = rxSliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(scratchbuff) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(
                static_cast<char *>(const_cast<void *>(sendbuff)) + redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            i == nranks - 2
                ? static_cast<void *>(
                      static_cast<char *>(recvbuff) + rxSliceOffsetInChunk +
                      r * baseRedSliceCount * typeSize +
                      std::min(r, (int)redSliceRemainder) * typeSize)
                : static_cast<void *>(static_cast<char *>(scratchbuff) +
                                      redOffset);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        if (i == nranks - 2) {
          runnerState->dagNodes[redNodeIdx].numChildren = 0;
        } else {
          runnerState->dagNodes[redNodeIdx].numChildren = 1;
        }
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     p2pNodeIdx));
        if (i != nranks - 2) {
          runnerState->dagNodes[redNodeIdx].children[0] =
              redSliceStartIdx + numRedSlices;
          TRACE(FLAGCX_UNIRUNNER, "rank %d redNode %d child 0: %d", rank,
                redNodeIdx, runnerState->dagNodes[redNodeIdx].children[0]);
        }
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank ReduceScatter topology (%d "
        "slices, %d redSlices)",
        nranks, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateTreeRed(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    void *scratchbuff, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t op, int root, flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  int algoRank = (rank - root + nranks) % nranks; // Rotate ranks so root is 0

  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      runnerState->dagNodes[0].nodeIdx = 0;
      runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeCpy;
      runnerState->dagNodes[0].nodeData.cpy.src = const_cast<void *>(sendbuff);
      runnerState->dagNodes[0].nodeData.cpy.dst = recvbuff;
      runnerState->dagNodes[0].nodeData.cpy.count = count;
      runnerState->dagNodes[0].nodeData.cpy.datatype = datatype;
      runnerState->dagNodes[0].numParents = 0;
      runnerState->dagNodes[0].numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[0]));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                              &runnerState->dagNodes[0]);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  if (runnerState->uniRunnerNRedSlices == 0) {
    if (count <= 0 || runnerState->uniRunnerRedSliceSize == 0) {
      runnerState->uniRunnerNRedSlices = 1;
    } else {
      runnerState->uniRunnerNRedSlices =
          ceil((float)count / comm->nranks / runnerState->uniRunnerNSlices /
               runnerState->uniRunnerRedSliceSize);
    }
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Nodes per slice chain:
  const int nTotalSteps = 8 * sizeof(int) - __builtin_clz(nranks - 1);
  int recvNodesPerSlice = algoRank ? __builtin_ctz(algoRank) : nTotalSteps;
  if (algoRank && recvNodesPerSlice &&
      nranks - algoRank <= (1 << (recvNodesPerSlice - 1))) {
    recvNodesPerSlice =
        nranks - algoRank - 1
            ? 8 * sizeof(int) - __builtin_clz(nranks - algoRank - 1)
            : 0;
    TRACE(FLAGCX_UNIRUNNER,
          "rank %d (algoRank %d) adjusted recvNodesPerSlice to %d from %d",
          rank, algoRank, recvNodesPerSlice, __builtin_ctz(algoRank));
  }
  const int sendNodesPerSlice = algoRank ? 1 : 0;
  const int redNodesPerSlice = recvNodesPerSlice * numRedSlices;
  const int nodesPerSlice =
      sendNodesPerSlice + recvNodesPerSlice + redNodesPerSlice;
  const int numNodes = nodesPerSlice * numSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d (algoRank %d) initUniRunnerStateTreeReduce called, count=%lu, "
        "numSlices=%d, numRedSlices=%d, recvSteps %d, sendSteps %d",
        comm->rank, algoRank, count, numSlices, numRedSlices, recvNodesPerSlice,
        sendNodesPerSlice);

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* halving doubling tree reduce
   * slice = s, step = i
   * recvNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices)
   * redNodeIdx = s * nodesPerSlice + i * (1 + numRedSlices) + 1..numRedSlices
   * sendNodeIdx = s * nodesPerSlice + recvNodesPerSlice + redNodesPerSlice
   */
  for (int s = 0; s < numSlices; s++) {
    size_t baseSliceCount = count / numSlices;
    size_t sliceRemainder = count % numSlices;
    size_t sliceCount = baseSliceCount + (s < sliceRemainder ? 1 : 0);
    size_t sliceOffset = s * baseSliceCount * typeSize;
    sliceOffset += std::min(s, (int)sliceRemainder) * typeSize;
    size_t rxOffset = count * typeSize + sliceOffset;

    TRACE(FLAGCX_UNIRUNNER,
          "Initializing rank %d (algoRank %d) slice %d, rxSliceCount %lu, "
          "rxSliceOffset %lu, rxOffset %lu",
          rank, algoRank, s, sliceCount, sliceOffset, rxOffset);

    // recv nodes and red nodes
    for (int i = 0; i < recvNodesPerSlice; i++) {
      int recvNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[recvNodeIdx].nodeIdx = recvNodeIdx;
      runnerState->dagNodes[recvNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps = 1;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops,
                       runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps *
                           sizeof(struct uniRunnerP2pOpData)));

      // Recv Node
      int peer = (rank + (1 << i)) % nranks;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].peerRank = peer;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].datatype =
          datatype;
      runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(scratchbuff) + rxOffset);
      TRACE(FLAGCX_UNIRUNNER,
            "rank %d (algoRank %d) recvNode %d recv from peer %d, count %lu, "
            "offset %lu",
            rank, algoRank, recvNodeIdx, peer, sliceCount, rxOffset);

      // Set up p2p node dependency
      if (recvNodeIdx == 0) {
        runnerState->dagNodes[recvNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[recvNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[recvNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[recvNodeIdx].numParents = 1 + numRedSlices;
        }
        runnerState->numPendingNodes++;
      }
      if (i == nTotalSteps - 1 && s == numSlices - 1) {
        runnerState->dagNodes[recvNodeIdx].numChildren = numRedSlices;
      } else {
        runnerState->dagNodes[recvNodeIdx].numChildren = 1 + numRedSlices;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[recvNodeIdx]));
      if (recvNodeIdx != 0) {
        int parentIdx = recvNodeIdx - nodesPerSlice;
        if (i > 0 && s == 0) {
          parentIdx =
              (numSlices - 1) * nodesPerSlice + (i - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[recvNodeIdx], 0,
                                     parentIdx));
        if (i > 0) {
          for (int r = 0; r < numRedSlices; r++) {
            FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[recvNodeIdx],
                                         r + 1,
                                         recvNodeIdx - numRedSlices + r));
          }
        }
      }
      for (int r = 0; r < numRedSlices; r++) {
        runnerState->dagNodes[recvNodeIdx].children[r] = recvNodeIdx + 1 + r;
      }
      if (s == numSlices - 1) {
        if (i != nTotalSteps - 1) {
          runnerState->dagNodes[recvNodeIdx].children[numRedSlices] =
              (i + 1) * (1 + numRedSlices);
        }
      } else {
        runnerState->dagNodes[recvNodeIdx].children[numRedSlices] =
            recvNodeIdx + nodesPerSlice;
      }

      // Reduce Node
      int redSliceStartIdx = globalNodeIdx;
      // Calculate redSliceCount with uneven distribution
      size_t baseRedSliceCount = sliceCount / numRedSlices;
      size_t redSliceRemainder = sliceCount % numRedSlices;
      for (int r = 0; r < numRedSlices; r++) {
        int redNodeIdx = globalNodeIdx++;
        runnerState->dagNodes[redNodeIdx].nodeIdx = redNodeIdx;
        runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
        runnerState->dagNodes[redNodeIdx].nodeData.red.triggerIdx = -1;
        // Calculate redCount and offset with uneven distribution
        size_t redCount = baseRedSliceCount;
        if (r < redSliceRemainder) {
          redCount++;
        }
        size_t redOffset = rxOffset + r * baseRedSliceCount * typeSize;
        // Add offset for all previous redSlices that got the remainder
        redOffset += std::min(r, (int)redSliceRemainder) * typeSize;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
            static_cast<void *>(static_cast<char *>(scratchbuff) + redOffset);
        void *redInput2Base =
            (i == 0) ? const_cast<void *>(sendbuff) : scratchbuff;
        runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
            static_cast<void *>(static_cast<char *>(redInput2Base) + redOffset -
                                count * typeSize);
        void *redOutput = (i == nTotalSteps - 1) ? recvbuff : scratchbuff;
        runnerState->dagNodes[redNodeIdx].nodeData.red.output =
            static_cast<void *>(static_cast<char *>(redOutput) + redOffset -
                                count * typeSize);
        runnerState->dagNodes[redNodeIdx].nodeData.red.count = redCount;
        runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
            runnerState->uniRunnerNThreads;
        runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

        // Set up red node dependency
        runnerState->numPendingNodes++;
        runnerState->dagNodes[redNodeIdx].numParents = 1;
        if (i == nTotalSteps - 1) {
          runnerState->dagNodes[redNodeIdx].numChildren = 0;
        } else {
          runnerState->dagNodes[redNodeIdx].numChildren = 1;
        }
        FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[redNodeIdx]));
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[redNodeIdx], 0,
                                     recvNodeIdx));
        if (i != nTotalSteps - 1) {
          runnerState->dagNodes[redNodeIdx].children[0] =
              redSliceStartIdx + numRedSlices;
        }
      }
    }

    // Send Node
    if (algoRank) {
      int sendNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[sendNodeIdx].nodeIdx = sendNodeIdx;
      runnerState->dagNodes[sendNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps = 1;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops,
                       runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps *
                           sizeof(struct uniRunnerP2pOpData)));

      int peer = (rank - (1 << (__builtin_ctz(algoRank))) + nranks) % nranks;

      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].peerRank = peer;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].datatype =
          datatype;
      runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(
              static_cast<char *>(recvNodesPerSlice == 0
                                      ? const_cast<void *>(sendbuff)
                                      : scratchbuff) +
              sliceOffset);
      TRACE(FLAGCX_UNIRUNNER,
            "rank %d (algoRank %d) sendNode %d send to peer %d, count %lu, "
            "offset %lu",
            rank, algoRank, sendNodeIdx, peer, sliceCount, sliceOffset);
      // Set up p2p node dependency
      if (recvNodesPerSlice == 0) {
        if (s == 0) {
          runnerState->dagNodes[sendNodeIdx].numParents = 0;
        } else {
          runnerState->dagNodes[sendNodeIdx].numParents = 1;
          runnerState->numPendingNodes++;
        }
      } else {
        runnerState->dagNodes[sendNodeIdx].numParents = 1 + numRedSlices;
        runnerState->numPendingNodes++;
      }
      if (s == numSlices - 1) {
        runnerState->dagNodes[sendNodeIdx].numChildren = 0;

      } else {
        runnerState->dagNodes[sendNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(allocDagNodeDeps(&runnerState->dagNodes[sendNodeIdx]));
      if (recvNodesPerSlice == 0) {
        if (s == 0) {
          flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                  &runnerState->dagNodes[sendNodeIdx]);
        } else {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx], 0,
                                       sendNodeIdx - nodesPerSlice));
        }
      } else {
        int parentIdx = sendNodeIdx - nodesPerSlice;
        if (s == 0) {
          parentIdx = (numSlices - 1) * nodesPerSlice +
                      (recvNodesPerSlice - 1) * (1 + numRedSlices);
        }
        FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx], 0,
                                     parentIdx));
        for (int r = 0; r < numRedSlices; r++) {
          FLAGCXCHECK(setDagNodeParent(&runnerState->dagNodes[sendNodeIdx],
                                       r + 1, sendNodeIdx - numRedSlices + r));
        }
      }
      if (s != numSlices - 1) {
        runnerState->dagNodes[sendNodeIdx].children[0] =
            sendNodeIdx + nodesPerSlice;
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Reduce (root %d) topology (%d "
        "slices, %d redSlices)",
        nranks, root, numSlices, numRedSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(
        FLAGCX_UNIRUNNER, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
        (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p) ? "P2P"
                                                                       : "RED",
        runnerState->dagNodes[i].numParents,
        runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_UNIRUNNER, "%s", childStr.c_str());
    }
  }

  return validateDagNodes(runnerState);
}

static void resetDagSchedulerRuntimeState(flagcxUniRunnerState *runnerState) {
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  runnerState->numPendingNodes = 0;
}

static flagcxResult_t captureUniRunnerDagTemplateFromState(
    const flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings,
    std::shared_ptr<uniRunnerDagTemplate> *dagTemplate) {
  std::shared_ptr<uniRunnerDagTemplate> captured =
      std::make_shared<uniRunnerDagTemplate>();
  captured->key = key;
  captured->hashValue = getUniRunnerDagPatternHash(key);
  captured->nodes.reserve(runnerState->numDagNodes);

  for (int i = 0; i < runnerState->numDagNodes; i++) {
    const uniRunnerDagNode *node = &runnerState->dagNodes[i];
    uniRunnerDagNodeDesc nodeDesc;
    nodeDesc.nodeType = node->nodeType;
    nodeDesc.nodeIdx = node->nodeIdx;
    if (node->numParents > 0) {
      nodeDesc.parents.assign(node->parents, node->parents + node->numParents);
    }
    if (node->numChildren > 0) {
      nodeDesc.children.assign(node->children,
                               node->children + node->numChildren);
    }

    if (node->nodeType == uniRunnerDagNodeTypeP2p) {
      for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerP2pOpData *op = &node->nodeData.p2p.ops[opIdx];
        uniRunnerDagP2pOpDesc opDesc;
        opDesc.count = op->count;
        opDesc.peerRank = op->peerRank;
        opDesc.datatype = op->datatype;
        opDesc.type = op->type;
        FLAGCXCHECK(captureBufferRef(op->addr, bindings, &opDesc.buffer));
        nodeDesc.p2pOps.push_back(opDesc);
      }
    } else if (node->nodeType == uniRunnerDagNodeTypeRed) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input1, bindings,
                                   &nodeDesc.red.input1));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.input2, bindings,
                                   &nodeDesc.red.input2));
      FLAGCXCHECK(captureBufferRef(node->nodeData.red.output, bindings,
                                   &nodeDesc.red.output));
      nodeDesc.red.count = node->nodeData.red.count;
      nodeDesc.red.nthreads = node->nodeData.red.nthreads;
      nodeDesc.red.datatype = node->nodeData.red.datatype;
      nodeDesc.red.redOp = node->nodeData.red.redOp;
    } else if (node->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.src, bindings,
                                   &nodeDesc.cpy.src));
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.dst, bindings,
                                   &nodeDesc.cpy.dst));
      nodeDesc.cpy.count = node->nodeData.cpy.count;
      nodeDesc.cpy.datatype = node->nodeData.cpy.datatype;
    } else if (node->nodeType == uniRunnerDagNodeTypeDevApi) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.devapi.sendbuff, bindings,
                                   &nodeDesc.devapi.sendbuff));
      FLAGCXCHECK(captureBufferRef(node->nodeData.devapi.recvbuff, bindings,
                                   &nodeDesc.devapi.recvbuff));
      FLAGCXCHECK(captureBufferRef(node->nodeData.devapi.flag, bindings,
                                   &nodeDesc.devapi.flag));
      nodeDesc.devapi.count = node->nodeData.devapi.count;
      nodeDesc.devapi.nthreads = node->nodeData.devapi.nthreads;
      nodeDesc.devapi.peerRank = node->nodeData.devapi.peerRank;
      nodeDesc.devapi.datatype = node->nodeData.devapi.datatype;
      nodeDesc.devapi.type = node->nodeData.devapi.type;
    } else {
      return flagcxNotSupported;
    }

    captured->nodes.push_back(nodeDesc);
  }

  *dagTemplate = captured;
  return flagcxSuccess;
}

static flagcxResult_t
materializeUniRunnerDagTemplate(flagcxUniRunnerState *runnerState,
                                const uniRunnerDagTemplate &dagTemplate,
                                const uniRunnerDagRuntimeBindings &bindings) {
  resetDagSchedulerRuntimeState(runnerState);
  runnerState->numDagNodes = static_cast<int>(dagTemplate.nodes.size());
  if (runnerState->numDagNodes == 0) {
    runnerState->dagNodes = NULL;
    return flagcxSuccess;
  }

  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int i = 0; i < runnerState->numDagNodes; i++) {
    const uniRunnerDagNodeDesc &src = dagTemplate.nodes[i];
    uniRunnerDagNode *dst = &runnerState->dagNodes[i];
    dst->nodeType = src.nodeType;
    dst->nodeIdx = src.nodeIdx;
    dst->numParents = static_cast<int>(src.parents.size());
    dst->numChildren = static_cast<int>(src.children.size());
    FLAGCXCHECK(allocDagNodeDeps(dst));
    dst->pendingParents = dst->numParents;
    for (int parentIdx = 0; parentIdx < dst->numParents; parentIdx++) {
      dst->parents[parentIdx] = src.parents[parentIdx];
    }
    for (int childIdx = 0; childIdx < dst->numChildren; childIdx++) {
      dst->children[childIdx] = src.children[childIdx];
    }

    if (dst->nodeType == uniRunnerDagNodeTypeP2p) {
      dst->nodeData.p2p.numOps = static_cast<int>(src.p2pOps.size());
      if (dst->nodeData.p2p.numOps > 0) {
        FLAGCXCHECK(flagcxCalloc(&dst->nodeData.p2p.ops,
                                 dst->nodeData.p2p.numOps *
                                     sizeof(struct uniRunnerP2pOpData)));
      }
      for (int opIdx = 0; opIdx < dst->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerDagP2pOpDesc &srcOp = src.p2pOps[opIdx];
        uniRunnerP2pOpData *dstOp = &dst->nodeData.p2p.ops[opIdx];
        dstOp->count = srcOp.count;
        dstOp->peerRank = srcOp.peerRank;
        dstOp->datatype = srcOp.datatype;
        dstOp->type = srcOp.type;
        FLAGCXCHECK(resolveBufferRef(srcOp.buffer, bindings, &dstOp->addr));
      }
    } else if (dst->nodeType == uniRunnerDagNodeTypeRed) {
      dst->nodeData.red.triggerIdx = -1;
      FLAGCXCHECK(resolveBufferRef(src.red.input1, bindings,
                                   &dst->nodeData.red.input1));
      FLAGCXCHECK(resolveBufferRef(src.red.input2, bindings,
                                   &dst->nodeData.red.input2));
      FLAGCXCHECK(resolveBufferRef(src.red.output, bindings,
                                   &dst->nodeData.red.output));
      dst->nodeData.red.count = src.red.count;
      dst->nodeData.red.nthreads = src.red.nthreads;
      dst->nodeData.red.datatype = src.red.datatype;
      dst->nodeData.red.redOp = src.red.redOp;
    } else if (dst->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.src, bindings, &dst->nodeData.cpy.src));
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.dst, bindings, &dst->nodeData.cpy.dst));
      dst->nodeData.cpy.count = src.cpy.count;
      dst->nodeData.cpy.datatype = src.cpy.datatype;
    } else if (dst->nodeType == uniRunnerDagNodeTypeDevApi) {
      dst->nodeData.devapi.triggerIdx = -1;
      FLAGCXCHECK(resolveBufferRef(src.devapi.sendbuff, bindings,
                                   &dst->nodeData.devapi.sendbuff));
      FLAGCXCHECK(resolveBufferRef(src.devapi.recvbuff, bindings,
                                   &dst->nodeData.devapi.recvbuff));
      FLAGCXCHECK(resolveBufferRef(src.devapi.flag, bindings,
                                   &dst->nodeData.devapi.flag));
      dst->nodeData.devapi.count = src.devapi.count;
      dst->nodeData.devapi.nthreads = src.devapi.nthreads;
      dst->nodeData.devapi.peerRank = src.devapi.peerRank;
      dst->nodeData.devapi.datatype = src.devapi.datatype;
      dst->nodeData.devapi.type = src.devapi.type;
    } else {
      return flagcxNotSupported;
    }

    if (dst->numParents == 0) {
      if (dst->nodeType == uniRunnerDagNodeTypeRed ||
          dst->nodeType == uniRunnerDagNodeTypeDevApi) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, dst);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, dst);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }

  return flagcxSuccess;
}

static flagcxResult_t tryLoadCachedUniRunnerDag(
    flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings, bool *cacheHit) {
  *cacheHit = false;
  std::shared_ptr<uniRunnerDagTemplate> dagTemplate;
  size_t hashValue = getUniRunnerDagPatternHash(key);
  if (!findUniRunnerDagTemplate(hashValue, key, &dagTemplate)) {
    flagcxResult_t loadRes = loadUniRunnerDagFromCacheDir(hashValue, key.rank);
    if (!findUniRunnerDagTemplate(hashValue, key, &dagTemplate)) {
      if (loadRes != flagcxSuccess) {
        return loadRes;
      }
      TRACE(FLAGCX_UNIRUNNER,
            "uniRunner DAG cache miss, algo=%s commOp=%s hash=%lu",
            uniRunnerDagAlgoTypeToString(key.algoType),
            uniRunnerCommOpToString(key.commOp), hashValue);
      return flagcxSuccess;
    }
  }

  TRACE(FLAGCX_UNIRUNNER, "uniRunner DAG cache hit, algo=%s commOp=%s hash=%lu",
        uniRunnerDagAlgoTypeToString(key.algoType),
        uniRunnerCommOpToString(key.commOp), hashValue);
  FLAGCXCHECK(
      materializeUniRunnerDagTemplate(runnerState, *dagTemplate, bindings));
  FLAGCXCHECK(validateDagNodes(runnerState));
  *cacheHit = true;
  return flagcxSuccess;
}

static flagcxResult_t
cacheBuiltUniRunnerDag(const flagcxUniRunnerState *runnerState,
                       const uniRunnerDagCacheKey &key,
                       const uniRunnerDagRuntimeBindings &bindings) {
  size_t hashValue = getUniRunnerDagPatternHash(key);
  std::shared_ptr<uniRunnerDagTemplate> dagTemplate;
  flagcxResult_t captureRes = captureUniRunnerDagTemplateFromState(
      runnerState, key, bindings, &dagTemplate);
  if (captureRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG capture skipped for hash=%lu, result=%d", hashValue,
          captureRes);
    return flagcxSuccess;
  }

  flagcxResult_t cacheRes = cacheUniRunnerDagTemplate(dagTemplate);
  if (cacheRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG cache persist skipped for hash=%lu, result=%d",
          hashValue, cacheRes);
  }
  return flagcxSuccess;
}

// Clean up DAG nodes
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_UNIRUNNER, "cleanupDagScheduler called");
  if (!runnerState) {
    return flagcxSuccess;
  }
  if (runnerState->dagNodes) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
          runnerState->dagNodes[i].nodeData.p2p.ops) {
        free(runnerState->dagNodes[i].nodeData.p2p.ops);
      }
      if (runnerState->dagNodes[i].parents) {
        free(runnerState->dagNodes[i].parents);
      }
      if (runnerState->dagNodes[i].children) {
        free(runnerState->dagNodes[i].children);
      }
    }
    free(runnerState->dagNodes);
    runnerState->dagNodes = NULL;
  }
  runnerState->numDagNodes = 0;
  runnerState->numPendingNodes = 0;
  return flagcxSuccess;
}

template <typename BuildFn>
static flagcxResult_t initUniRunnerStateCached(
    flagcxUniRunnerState *runnerState, const uniRunnerDagCacheKey &key,
    const uniRunnerDagRuntimeBindings &bindings, BuildFn buildFn) {
  bool cacheHit = false;
  flagcxResult_t cacheLoadRes =
      tryLoadCachedUniRunnerDag(runnerState, key, bindings, &cacheHit);
  if (cacheLoadRes == flagcxSuccess && cacheHit) {
    return flagcxSuccess;
  }
  if (cacheLoadRes != flagcxSuccess) {
    TRACE(FLAGCX_UNIRUNNER,
          "uniRunner DAG cache load failed, rebuild dag instead, result=%d",
          cacheLoadRes);
    (void)cleanupDagScheduler(runnerState);
    resetDagSchedulerRuntimeState(runnerState);
  }

  FLAGCXCHECK(buildFn());
  return cacheBuiltUniRunnerDag(runnerState, key, bindings);
}

flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoLocRed, flagcxCommOpAllReduce, count, datatype, op, -1, 0,
      runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateLocRed(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateGroupedAG(flagcxUniRunnerState *runnerState,
                                           const void *sendbuff, void *recvbuff,
                                           size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxComm_t comm, int groupSize) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * comm->nranks * typeSize;

  uniRunnerDagCacheKey key =
      makeUniRunnerDagCacheKey(uniRunnerDagAlgoGroupedAG, flagcxCommOpAllGather,
                               count, datatype, flagcxRedNoOp, -1, groupSize,
                               runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff, count,
                                        datatype, comm, groupSize);
  });
}

flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAG, flagcxCommOpAllReduce, count, datatype,
      flagcxRedNoOp, -1, 0, runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAG(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAR, flagcxCommOpAllReduce, count, datatype, op, -1, 0,
      runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAR(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoSlicedAR, flagcxCommOpAllReduce, count, datatype, op, -1,
      0, runnerState, comm, sendbuff, recvbuff, NULL);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateDevApiTest(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm) {
  return buildUniRunnerStateDevApiTest(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
}

flagcxResult_t initUniRunnerStateDevARPut(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm) {
  return buildUniRunnerStateDevAR(runnerState, sendbuff, recvbuff, count,
                                  datatype, op, comm, false);
}

flagcxResult_t initUniRunnerStateDevARGet(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm) {
  return buildUniRunnerStateDevAR(runnerState, sendbuff, recvbuff, count,
                                  datatype, op, comm, true);
}

flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * comm->nranks * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = count * comm->nranks * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingRS, flagcxCommOpReduceScatter, count, datatype, op,
      -1, 0, runnerState, comm, sendbuff, recvbuff, scratchbuff);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                     scratchbuff, count, datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateTreeRed(flagcxUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         void *scratchbuff, size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, int root,
                                         flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = 2 * count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoTreeRed, flagcxCommOpReduce, count, datatype, op, root, 0,
      runnerState, comm, sendbuff, recvbuff, scratchbuff);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                      scratchbuff, count, datatype, op, root,
                                      comm);
  });
}

static flagcxResult_t launchP2pOps(flagcxUniRunnerState *runnerState,
                                   flagcxHeteroComm_t comm) {
  // Dequeue
  uniRunnerDagNode *current =
      flagcxIntruQueueDequeue(&runnerState->p2pReadyQueue);
  void *flag = getDagNodeFlag(runnerState, current->nodeIdx);
  flagcxStream_t currentStream =
      getDagNodeExecutionStream(runnerState, current);

  if (current->nodeType == uniRunnerDagNodeTypeP2p) {
    // Mark the node as submitted before wiring its completion dependency.
    TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWrite flag %d: PEND",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->commStream, flag,
                                                  flagcxStreamFlagPend, 0));
    for (int i = 0; i < current->numParents; i++) {
      int parentIdx = current->parents[i];
      uniRunnerDagNode *parent = &runnerState->dagNodes[parentIdx];
      if (getDagNodeExecutionStream(runnerState, parent) == currentStream) {
        TRACE(FLAGCX_UNIRUNNER,
              "rank %d p2p op %d skip same-stream wait for parent %d",
              comm->rank, current->nodeIdx, parentIdx);
        continue;
      }
      void *parentFlag = getDagNodeFlag(runnerState, parentIdx);
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          runnerState->commStream, parentFlag, flagcxStreamFlagDone, 0));
      TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWait flag %d: DONE",
            comm->rank, current->nodeIdx, parentIdx);
    }

    // Prepare ops list
    struct uniRunnerP2pOpData *ops = current->nodeData.p2p.ops;

    // Start Group P2P
    FLAGCXCHECK(flagcxHeteroGroupStart());
    for (int i = 0; i < current->nodeData.p2p.numOps; i++) {
      struct uniRunnerP2pOpData *op = &ops[i];
      if (op->type == flagcxDevicePrimSend) {
        FLAGCXCHECK(flagcxHeteroSend(op->addr, op->count, op->datatype,
                                     op->peerRank, comm,
                                     runnerState->commStream));
      } else if (op->type == flagcxDevicePrimRecv) {
        FLAGCXCHECK(flagcxHeteroRecv(op->addr, op->count, op->datatype,
                                     op->peerRank, comm,
                                     runnerState->commStream));
      }
    }
    FLAGCXCHECK(flagcxHeteroGroupEnd());

    TRACE(FLAGCX_UNIRUNNER, "rank %d p2p op %d streamWrite flag %d: DONE",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->commStream, flag,
                                                  flagcxStreamFlagDone, 0));
  } else if (current->nodeType == uniRunnerDagNodeTypeCpy) {
    TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWrite flag %d: PEND",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->cpyStream, flag,
                                                  flagcxStreamFlagPend, 0));
    for (int i = 0; i < current->numParents; i++) {
      int parentIdx = current->parents[i];
      uniRunnerDagNode *parent = &runnerState->dagNodes[parentIdx];
      if (getDagNodeExecutionStream(runnerState, parent) == currentStream) {
        TRACE(FLAGCX_UNIRUNNER,
              "rank %d cpy op %d skip same-stream wait for parent %d",
              comm->rank, current->nodeIdx, parentIdx);
        continue;
      }
      void *parentFlag = getDagNodeFlag(runnerState, parentIdx);
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          runnerState->cpyStream, parentFlag, flagcxStreamFlagDone, 0));
      TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWait flag %d: DONE",
            comm->rank, current->nodeIdx, parentIdx);
    }

    if (current->nodeData.cpy.waitFlag != NULL) {
      FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
          runnerState->cpyStream, current->nodeData.cpy.waitFlag,
          flagcxStreamFlagDone, 0));
      TRACE(FLAGCX_UNIRUNNER,
            "rank %d cpy op %d streamWait ipc flag %p: DONE", comm->rank,
            current->nodeIdx, current->nodeData.cpy.waitFlag);
    }

    // Launch copy when this cpy node carries a real transfer. Some DevAR
    // nodes use the copy stream only as a fan-in or IPC flag wait.
    if (current->nodeData.cpy.count != 0 && current->nodeData.cpy.src != NULL &&
        current->nodeData.cpy.dst != NULL) {
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          current->nodeData.cpy.dst, current->nodeData.cpy.src,
          current->nodeData.cpy.count *
              getFlagcxDataTypeSize(current->nodeData.cpy.datatype),
          flagcxMemcpyDeviceToDevice, runnerState->cpyStream, NULL));
    }

    // Write flag to stream
    TRACE(FLAGCX_UNIRUNNER, "rank %d cpy op %d streamWrite flag %d: DONE",
          comm->rank, current->nodeIdx, current->nodeIdx);
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(runnerState->cpyStream, flag,
                                                  flagcxStreamFlagDone, 0));
  } else {
    return flagcxSystemError;
  }

  return flagcxSuccess;
}

static flagcxResult_t enqueueReadyQueue(flagcxUniRunnerState *runnerState,
                                        int nodeIdx) {
  if (runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeP2p ||
      runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeCpy) {
    flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else if (runnerState->dagNodes[nodeIdx].nodeType ==
                 uniRunnerDagNodeTypeRed ||
             runnerState->dagNodes[nodeIdx].nodeType ==
                 uniRunnerDagNodeTypeDevApi) {
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else {
    return flagcxNotSupported;
  }
  runnerState->numPendingNodes--;
  return flagcxSuccess;
}

static flagcxResult_t notifyChildrenScheduled(flagcxUniRunnerState *runnerState,
                                              uniRunnerDagNode *current) {
  for (int i = 0; i < current->numChildren; i++) {
    uniRunnerDagNode *child = &runnerState->dagNodes[current->children[i]];
    if (child->pendingParents <= 0) {
      return flagcxInternalError;
    }
    child->pendingParents--;
    if (child->pendingParents == 0) {
      FLAGCXCHECK(enqueueReadyQueue(runnerState, current->children[i]));
    }
  }
  return flagcxSuccess;
}

// Process ready queue: submit ready nodes to the corresponding execution
// stream/FIFO. Child readiness is host-scheduled immediately after submission;
// same-stream execution dependencies rely on launch order, while cross-stream
// dependencies are enforced via stream flags.
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  // process p2pReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue)) {
    uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->p2pReadyQueue);
    FLAGCXCHECK(launchP2pOps(runnerState, comm));
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  // process redReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->redReadyQueue)) {
    struct uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->redReadyQueue);
    uint64_t flagIn =
        current->numParents == 0
            ? 0
            : (uintptr_t)getDagNodeFlag(runnerState, current->parents[0]);
    uint64_t flagOut = (uintptr_t)getDagNodeFlag(runnerState, current->nodeIdx);
    // The current FIFO-backed nodes have at most one parent. Multi-parent
    // dependencies are handled for P2P/CPY nodes by emitting one stream wait
    // per parent; FIFO-backed nodes would need an explicit fan-in flag if that
    // ever changes.
    if (current->numParents > 1) {
      return flagcxInvalidArgument;
    }
    int idx = -1;
    if (current->nodeType == uniRunnerDagNodeTypeRed) {
      FLAGCXCHECK(enqueue(
          (void *)runnerState->fifo->buffer,
          (uintptr_t)current->nodeData.red.input1,
          (uintptr_t)current->nodeData.red.input2,
          (uintptr_t)current->nodeData.red.output, current->nodeData.red.count,
          current->nodeData.red.nthreads, current->nodeData.red.datatype,
          current->nodeData.red.redOp, flagIn, flagOut, &idx));
    } else if (current->nodeType == uniRunnerDagNodeTypeDevApi) {
      FLAGCXCHECK(enqueueDevApi(
          (void *)runnerState->fifo->buffer,
          (uintptr_t)current->nodeData.devapi.sendbuff,
          (uintptr_t)current->nodeData.devapi.recvbuff,
          (uintptr_t)current->nodeData.devapi.flag,
          current->nodeData.devapi.count, current->nodeData.devapi.nthreads,
          current->nodeData.devapi.datatype, current->nodeData.devapi.type,
          flagIn, flagOut, &idx));
    } else {
      return flagcxNotSupported;
    }
    if (idx == -1) {
      sched_yield();
      break; // FIFO full, skip for now
    }
    // Dequeue
    flagcxIntruQueueDequeue(&runnerState->redReadyQueue);
    if (current->nodeType == uniRunnerDagNodeTypeRed) {
      current->nodeData.red.triggerIdx = idx;
    } else {
      current->nodeData.devapi.triggerIdx = idx;
    }
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  return flagcxSuccess;
}

flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  runnerState->dagNodes = NULL;
  runnerState->numDagNodes = 0;
  runnerState->streamFlagsSize = 0;

  runnerState->uniRunnerNSlices = flagcxParamUniRunnerNSlices();
  runnerState->uniRunnerNThreads = flagcxParamUniRunnerNThreads();
  runnerState->uniRunnerNBlocks = flagcxParamUniRunnerNBlocks();
  runnerState->uniRunnerNRedSlices = flagcxParamUniRunnerNRedSlices();
  runnerState->uniRunnerRedSliceSize = flagcxParamUniRunnerRedSliceSize();

  // Set device context
  FLAGCXCHECK(deviceAdaptor->setDevice(hcomm->cudaDev));

  // Create FIFO
  runnerState->fifo = new flagcxFifo();
  FLAGCXCHECK(runnerState->fifo->flagcxRedFifoInit());
  // hcomm->proxyState->uniRunnerState.fifo->buffer is the host pointer
  // hcomm->uniRunnerFifoBuffer stores the device pointer to fifo buffer
  FLAGCXCHECK(deviceAdaptor->hostGetDevicePointer(
      &hcomm->uniRunnerFifoBuffer, (void *)runnerState->fifo->buffer));

  // Initialize queues
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  runnerState->numPendingNodes = 0;

  // Create dedicated reduce and copy streams
  flagcxStream_t redStream;
  FLAGCXCHECK(deviceAdaptor->streamCreate(&redStream));
  flagcxStream_t cpyStream;
  FLAGCXCHECK(deviceAdaptor->streamCreate(&cpyStream));
  runnerState->redStream = redStream;
  runnerState->cpyStream = cpyStream;
  runnerState->commStream = stream;
  return flagcxSuccess;
}

flagcxResult_t cleanupUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxStream_t commStream = hcomm->proxyState->uniRunnerState.commStream;
  flagcxStream_t redStream = hcomm->proxyState->uniRunnerState.redStream;
  flagcxStream_t cpyStream = hcomm->proxyState->uniRunnerState.cpyStream;

  // Clean up DAG scheduler
  FLAGCXCHECK(cleanupDagScheduler(&hcomm->proxyState->uniRunnerState));

  // Outstanding stream waits/writes may still touch streamFlags when
  // runUniRunner exits early on an error path, so synchronize before marking
  // the reusable flag-address queue inactive for this invocation.
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(redStream));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(cpyStream));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(commStream));

  FLAGCXCHECK(destroyUniRunnerDevApiDataViews(
      &hcomm->proxyState->uniRunnerState));
  hcomm->proxyState->uniRunnerState.streamFlagsSize = 0;

  // Destroy streams
  FLAGCXCHECK(deviceAdaptor->streamDestroy(redStream));
  FLAGCXCHECK(deviceAdaptor->streamDestroy(cpyStream));

  // Destroy fifo
  FLAGCXCHECK(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
  delete hcomm->proxyState->uniRunnerState.fifo;
  hcomm->uniRunnerFifoBuffer = NULL;

  return flagcxSuccess;
}

flagcxResult_t runUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxFifo_t fifo = hcomm->proxyState->uniRunnerState.fifo;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  TRACE(FLAGCX_UNIRUNNER, "runUniRunner called");
  FLAGCXCHECK(prepareDagStreamFlags(runnerState));

#ifdef COMPILE_KERNEL_HOST
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(
      hcomm->uniRunnerFifoBuffer, runnerState->uniRunnerNThreads,
      runnerState->uniRunnerNBlocks, runnerState->redStream);
#endif

  // Main scheduling loop using DAG-based queue scheduling
  while (true) {
    if (flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->redReadyQueue) &&
        runnerState->numPendingNodes == 0) {
      TRACE(
          FLAGCX_UNIRUNNER,
          "runUniRunner: all submitted work drained, terminating runner loop");
      __atomic_store_n(fifo->buffer + flagcxFifoIdxTerminate, 1,
                       __ATOMIC_RELEASE);
      break;
    }

    FLAGCXCHECK(processReadyQueue(runnerState, hcomm));
  }
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(runnerState->redStream));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(runnerState->cpyStream));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(runnerState->commStream));

  return flagcxSuccess;
}

flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState) {
  FLAGCXCHECK(cleanupDagScheduler(runnerState));
  FLAGCXCHECK(destroyUniRunnerDevApiState(runnerState));
  return destroyStreamFlagQueue(runnerState);
}
