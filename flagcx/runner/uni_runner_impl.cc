#include "uni_runner_impl.h"
#include "adaptor.h"
#include "comm.h"
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
constexpr size_t kUniRunnerRedCounterStrideBytes = 128;

struct uniRunnerDagRuntimeBindings {
  const void *inputBase = NULL;
  void *outputBase = NULL;
  void *scratchBase = NULL;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t scratchBytes = 0;
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
  return lhs.formatVersion == rhs.formatVersion &&
         lhs.algoType == rhs.algoType && lhs.commOp == rhs.commOp &&
         lhs.count == rhs.count && lhs.datatype == rhs.datatype &&
         lhs.redOp == rhs.redOp && lhs.rank == rhs.rank &&
         lhs.nranks == rhs.nranks && lhs.root == rhs.root;
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
    flagcxDataType_t datatype, flagcxRedOp_t redOp, int root,
    flagcxComm_t comm) {
  uniRunnerDagCacheKey key{};
  key.formatVersion = kUniRunnerDagCacheFormatVersion;
  key.algoType = algoType;
  key.commOp = commOp;
  key.count = count;
  key.datatype = datatype;
  key.redOp = redOp;
  key.rank = comm->rank;
  key.nranks = comm->nranks;
  key.root = root;
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
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.formatVersion));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.algoType));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.commOp));
  hashValue = hashCombine(hashValue, key.count);
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.datatype));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redOp));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.rank));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nranks));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.root + 1));
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

static flagcxResult_t resolveUniRunnerRedWorkerCount(
    const flagcxUniRunnerState *runnerState, size_t count,
    uint32_t *workerCount) {
  if (runnerState->uniRunnerNBlocks == 0 || count > UINT32_MAX) {
    return flagcxInvalidArgument;
  }
  if (!runnerState->redSliceWorkersEnabled) {
    *workerCount = 1;
    return flagcxSuccess;
  }
  if (runnerState->uniRunnerNRedSlices == 0) {
    return flagcxInvalidArgument;
  }
  if (count == 0) {
    *workerCount = 1;
    return flagcxSuccess;
  }
  uint64_t resolved = runnerState->uniRunnerNRedSlices;
  if (runnerState->uniRunnerNBlocks < resolved) {
    resolved = runnerState->uniRunnerNBlocks;
  }
  if (count < resolved) {
    resolved = count;
  }
  if (resolved < 1 || resolved > UINT32_MAX) {
    return flagcxInvalidArgument;
  }
  *workerCount = static_cast<uint32_t>(resolved);
  return flagcxSuccess;
}

static flagcxResult_t ensureRedCompletionCounterCapacity(
    flagcxUniRunnerState *runnerState, size_t required) {
  if (required <= runnerState->redCompletionCountersCapacity) {
    return flagcxSuccess;
  }
  size_t newCapacity = runnerState->redCompletionCountersCapacity == 0
                           ? 1
                           : runnerState->redCompletionCountersCapacity;
  while (newCapacity < required) {
    if (newCapacity > SIZE_MAX / 2) {
      return flagcxInvalidArgument;
    }
    newCapacity *= 2;
  }
  if (newCapacity > SIZE_MAX / kUniRunnerRedCounterStrideBytes) {
    return flagcxInvalidArgument;
  }
  void *newPool = NULL;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      &newPool, newCapacity * kUniRunnerRedCounterStrideBytes,
      flagcxMemDevice, NULL));
  if (runnerState->redCompletionCountersPool != NULL) {
    flagcxResult_t freeRes = deviceAdaptor->deviceFree(
        runnerState->redCompletionCountersPool, flagcxMemDevice, NULL);
    if (freeRes != flagcxSuccess) {
      (void)deviceAdaptor->deviceFree(newPool, flagcxMemDevice, NULL);
      return freeRes;
    }
  }
  runnerState->redCompletionCountersPool = newPool;
  runnerState->redCompletionCountersCapacity = newCapacity;
  return flagcxSuccess;
}

static flagcxResult_t
destroyRedCompletionCounterPool(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL) {
    return flagcxSuccess;
  }
  if (runnerState->redCompletionCountersPool != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(
        runnerState->redCompletionCountersPool, flagcxMemDevice, NULL));
    runnerState->redCompletionCountersPool = NULL;
  }
  runnerState->redCompletionCountersCapacity = 0;
  return flagcxSuccess;
}

static flagcxResult_t
prepareRedWorkerRuntimeState(flagcxUniRunnerState *runnerState) {
  size_t multiWorkerRedNodeCount = 0;
  for (int i = 0; i < runnerState->numDagNodes; ++i) {
    uniRunnerDagNode *node = &runnerState->dagNodes[i];
    if (node->nodeType != uniRunnerDagNodeTypeRed) {
      continue;
    }
    uniRunnerRedNodeData *red = &node->nodeData.red;
    red->triggerIdx = -1;
    red->nextWorkerToSubmit = 0;
    red->completionCounter = NULL;
    FLAGCXCHECK(resolveUniRunnerRedWorkerCount(runnerState, red->count,
                                               &red->workerCount));
    if (red->workerCount > 1) {
      multiWorkerRedNodeCount++;
    }
  }

  FLAGCXCHECK(ensureRedCompletionCounterCapacity(runnerState,
                                                 multiWorkerRedNodeCount));
  if (multiWorkerRedNodeCount == 0) {
    return flagcxSuccess;
  }
  if (multiWorkerRedNodeCount >
      SIZE_MAX / kUniRunnerRedCounterStrideBytes) {
    return flagcxInvalidArgument;
  }
  size_t bytes = multiWorkerRedNodeCount * kUniRunnerRedCounterStrideBytes;
  FLAGCXCHECK(deviceAdaptor->deviceMemset(
      runnerState->redCompletionCountersPool, 0, bytes, flagcxMemDevice,
      runnerState->redStream));

  char *poolBase =
      static_cast<char *>(runnerState->redCompletionCountersPool);
  size_t ordinal = 0;
  for (int i = 0; i < runnerState->numDagNodes; ++i) {
    uniRunnerDagNode *node = &runnerState->dagNodes[i];
    if (node->nodeType == uniRunnerDagNodeTypeRed &&
        node->nodeData.red.workerCount > 1) {
      node->nodeData.red.completionCounter = reinterpret_cast<uint32_t *>(
          poolBase + ordinal * kUniRunnerRedCounterStrideBytes);
      ordinal++;
    }
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

static inline flagcxStream_t
getDagNodeExecutionStream(flagcxUniRunnerState *runnerState,
                          const uniRunnerDagNode *node) {
  switch (node->nodeType) {
    case uniRunnerDagNodeTypeP2p:
      return runnerState->commStream;
    case uniRunnerDagNodeTypeRed:
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
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      uniRunnerDagNode *node = &runnerState->dagNodes[0];
      node->nodeIdx = 0;
      node->nodeType = uniRunnerDagNodeTypeCpy;
      node->nodeData.cpy.src = const_cast<void *>(sendbuff);
      node->nodeData.cpy.dst = recvbuff;
      node->nodeData.cpy.count = count;
      node->nodeData.cpy.datatype = datatype;
      node->numParents = 0;
      node->numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(node));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  int numSlices = runnerState->uniRunnerNSlices;
  int ringSteps = nranks - 1;
  int nodesPerSlice = 3 * ringSteps;
  int numNodes = numSlices * nodesPerSlice;
  auto rsP2p = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i;
  };
  auto rsRed = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i + 1;
  };
  auto agP2p = [nodesPerSlice, ringSteps](int s, int i) {
    return s * nodesPerSlice + 2 * ringSteps + i;
  };

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(
      &runnerState->dagNodes, numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  for (int s = 0; s < numSlices; ++s) {
    for (int i = 0; i < ringSteps; ++i) {
      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;
      size_t txRankCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankCount / numSlices;
      size_t rxBaseSliceCount = rxRankCount / numSlices;
      size_t txSliceRemainder = txRankCount % numSlices;
      size_t rxSliceRemainder = rxRankCount % numSlices;
      size_t txSliceCount =
          txBaseSliceCount + (s < (int)txSliceRemainder ? 1 : 0);
      size_t rxSliceCount =
          rxBaseSliceCount + (s < (int)rxSliceRemainder ? 1 : 0);
      size_t txSliceOffset =
          (s * txBaseSliceCount + std::min(s, (int)txSliceRemainder)) *
          typeSize;
      size_t rxSliceOffset =
          (s * rxBaseSliceCount + std::min(s, (int)rxSliceRemainder)) *
          typeSize;
      size_t txOffset =
          (txChunk * baseRankChunkCount +
           std::min(txChunk, (int)rankChunkRemainder)) *
              typeSize +
          txSliceOffset;
      size_t rxOffset =
          (rxChunk * baseRankChunkCount +
           std::min(rxChunk, (int)rankChunkRemainder)) *
              typeSize +
          rxSliceOffset;

      int p2pIdx = rsP2p(s, i);
      uniRunnerDagNode *p2p = &runnerState->dagNodes[p2pIdx];
      p2p->nodeIdx = p2pIdx;
      p2p->nodeType = uniRunnerDagNodeTypeP2p;
      p2p->nodeData.p2p.numOps = 2;
      FLAGCXCHECK(flagcxCalloc(&p2p->nodeData.p2p.ops,
                               2 * sizeof(struct uniRunnerP2pOpData)));
      void *sendBase = i == 0 ? const_cast<void *>(sendbuff) : recvbuff;
      p2p->nodeData.p2p.ops[0] =
          {static_cast<char *>(sendBase) + txOffset, txSliceCount, nextRank,
           datatype, flagcxDevicePrimSend};
      p2p->nodeData.p2p.ops[1] =
          {static_cast<char *>(recvbuff) + rxOffset, rxSliceCount, prevRank,
           datatype, flagcxDevicePrimRecv};
      bool hasCommParent = !(s == 0 && i == 0);
      p2p->numParents = (hasCommParent ? 1 : 0) + (i > 0 ? 1 : 0);
      p2p->numChildren = 2;
      FLAGCXCHECK(allocDagNodeDeps(p2p));
      int parentSlot = 0;
      if (hasCommParent) {
        FLAGCXCHECK(setDagNodeParent(
            p2p, parentSlot++,
            s > 0 ? rsP2p(s - 1, i) : rsP2p(numSlices - 1, i - 1)));
      }
      if (i > 0) {
        FLAGCXCHECK(setDagNodeParent(p2p, parentSlot, rsRed(s, i - 1)));
      }
      p2p->children[0] = rsRed(s, i);
      p2p->children[1] = s + 1 < numSlices
                             ? rsP2p(s + 1, i)
                             : (i + 1 < ringSteps ? rsP2p(0, i + 1)
                                                  : agP2p(0, 0));
      if (p2p->numParents == 0) {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, p2p);
      } else {
        runnerState->numPendingNodes++;
      }

      int redIdx = rsRed(s, i);
      uniRunnerDagNode *red = &runnerState->dagNodes[redIdx];
      red->nodeIdx = redIdx;
      red->nodeType = uniRunnerDagNodeTypeRed;
      red->nodeData.red.input1 = static_cast<char *>(recvbuff) + rxOffset;
      red->nodeData.red.input2 =
          static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset;
      red->nodeData.red.output = static_cast<char *>(recvbuff) + rxOffset;
      red->nodeData.red.count = rxSliceCount;
      red->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
      red->nodeData.red.datatype = datatype;
      red->nodeData.red.redOp = op;
      red->numParents = 1;
      red->numChildren = 1;
      FLAGCXCHECK(allocDagNodeDeps(red));
      FLAGCXCHECK(setDagNodeParent(red, 0, p2pIdx));
      red->children[0] =
          i + 1 < ringSteps ? rsP2p(s, i + 1) : agP2p(s, 0);
      runnerState->numPendingNodes++;
    }

    for (int i = 0; i < ringSteps; ++i) {
      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;
      size_t txRankCount =
          baseRankChunkCount + (txChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t rxRankCount =
          baseRankChunkCount + (rxChunk < (int)rankChunkRemainder ? 1 : 0);
      size_t txBaseSliceCount = txRankCount / numSlices;
      size_t rxBaseSliceCount = rxRankCount / numSlices;
      size_t txSliceRemainder = txRankCount % numSlices;
      size_t rxSliceRemainder = rxRankCount % numSlices;
      size_t txSliceCount =
          txBaseSliceCount + (s < (int)txSliceRemainder ? 1 : 0);
      size_t rxSliceCount =
          rxBaseSliceCount + (s < (int)rxSliceRemainder ? 1 : 0);
      size_t txOffset =
          (txChunk * baseRankChunkCount +
           std::min(txChunk, (int)rankChunkRemainder) +
           s * txBaseSliceCount + std::min(s, (int)txSliceRemainder)) *
          typeSize;
      size_t rxOffset =
          (rxChunk * baseRankChunkCount +
           std::min(rxChunk, (int)rankChunkRemainder) +
           s * rxBaseSliceCount + std::min(s, (int)rxSliceRemainder)) *
          typeSize;

      int p2pIdx = agP2p(s, i);
      uniRunnerDagNode *p2p = &runnerState->dagNodes[p2pIdx];
      p2p->nodeIdx = p2pIdx;
      p2p->nodeType = uniRunnerDagNodeTypeP2p;
      p2p->nodeData.p2p.numOps = 2;
      FLAGCXCHECK(flagcxCalloc(&p2p->nodeData.p2p.ops,
                               2 * sizeof(struct uniRunnerP2pOpData)));
      p2p->nodeData.p2p.ops[0] =
          {static_cast<char *>(recvbuff) + txOffset, txSliceCount, nextRank,
           datatype, flagcxDevicePrimSend};
      p2p->nodeData.p2p.ops[1] =
          {static_cast<char *>(recvbuff) + rxOffset, rxSliceCount, prevRank,
           datatype, flagcxDevicePrimRecv};
      p2p->numParents = i == 0 ? 2 : 1;
      bool hasCommChild = s + 1 < numSlices || i + 1 < ringSteps;
      p2p->numChildren = hasCommChild ? 1 : 0;
      FLAGCXCHECK(allocDagNodeDeps(p2p));
      int commParent = s > 0 ? agP2p(s - 1, i)
                             : (i > 0 ? agP2p(numSlices - 1, i - 1)
                                      : rsP2p(numSlices - 1, ringSteps - 1));
      FLAGCXCHECK(setDagNodeParent(p2p, 0, commParent));
      if (i == 0) {
        FLAGCXCHECK(setDagNodeParent(p2p, 1, rsRed(s, ringSteps - 1)));
      }
      if (hasCommChild) {
        p2p->children[0] = s + 1 < numSlices
                               ? agP2p(s + 1, i)
                               : agP2p(0, i + 1);
      }
      runnerState->numPendingNodes++;
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Sliced AllReduce topology "
        "(%d slices)",
        nranks, numSlices);
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
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      uniRunnerDagNode *node = &runnerState->dagNodes[0];
      node->nodeIdx = 0;
      node->nodeType = uniRunnerDagNodeTypeCpy;
      node->nodeData.cpy.src = const_cast<void *>(sendbuff);
      node->nodeData.cpy.dst = recvbuff;
      node->nodeData.cpy.count = count;
      node->nodeData.cpy.datatype = datatype;
      node->numParents = 0;
      node->numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(node));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  int numSlices = runnerState->uniRunnerNSlices;
  int ringSteps = nranks - 1;
  int nodesPerSlice = 2 * ringSteps;
  int numNodes = numSlices * nodesPerSlice;
  auto p2pIndex = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i;
  };
  auto redIndex = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i + 1;
  };

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(
      &runnerState->dagNodes, numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  for (int s = 0; s < numSlices; ++s) {
    for (int i = 0; i < ringSteps; ++i) {
      int txChunk = (rank - i - 1 + nranks) % nranks;
      int rxChunk = (rank - i - 2 + nranks) % nranks;
      size_t baseSliceCount = count / numSlices;
      size_t sliceRemainder = count % numSlices;
      size_t sliceCount =
          baseSliceCount + (s < (int)sliceRemainder ? 1 : 0);
      size_t sliceOffset =
          (s * baseSliceCount + std::min(s, (int)sliceRemainder)) * typeSize;
      size_t txOffset = txChunk * count * typeSize + sliceOffset;
      size_t rxOffset = rxChunk * count * typeSize + sliceOffset;

      int p2pIdx = p2pIndex(s, i);
      uniRunnerDagNode *p2p = &runnerState->dagNodes[p2pIdx];
      p2p->nodeIdx = p2pIdx;
      p2p->nodeType = uniRunnerDagNodeTypeP2p;
      p2p->nodeData.p2p.numOps = 2;
      FLAGCXCHECK(flagcxCalloc(&p2p->nodeData.p2p.ops,
                               2 * sizeof(struct uniRunnerP2pOpData)));
      void *sendBase = i == 0 ? const_cast<void *>(sendbuff) : scratchbuff;
      p2p->nodeData.p2p.ops[0] =
          {static_cast<char *>(sendBase) + txOffset, sliceCount, nextRank,
           datatype, flagcxDevicePrimSend};
      p2p->nodeData.p2p.ops[1] =
          {static_cast<char *>(scratchbuff) + rxOffset, sliceCount, prevRank,
           datatype, flagcxDevicePrimRecv};
      bool hasCommParent = !(s == 0 && i == 0);
      p2p->numParents = (hasCommParent ? 1 : 0) + (i > 0 ? 1 : 0);
      bool hasCommChild = s + 1 < numSlices || i + 1 < ringSteps;
      p2p->numChildren = 1 + (hasCommChild ? 1 : 0);
      FLAGCXCHECK(allocDagNodeDeps(p2p));
      int parentSlot = 0;
      if (hasCommParent) {
        FLAGCXCHECK(setDagNodeParent(
            p2p, parentSlot++, s > 0 ? p2pIndex(s - 1, i)
                                     : p2pIndex(numSlices - 1, i - 1)));
      }
      if (i > 0) {
        FLAGCXCHECK(setDagNodeParent(p2p, parentSlot, redIndex(s, i - 1)));
      }
      p2p->children[0] = redIndex(s, i);
      if (hasCommChild) {
        p2p->children[1] = s + 1 < numSlices
                               ? p2pIndex(s + 1, i)
                               : p2pIndex(0, i + 1);
      }
      if (p2p->numParents == 0) {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, p2p);
      } else {
        runnerState->numPendingNodes++;
      }

      int redIdx = redIndex(s, i);
      uniRunnerDagNode *red = &runnerState->dagNodes[redIdx];
      red->nodeIdx = redIdx;
      red->nodeType = uniRunnerDagNodeTypeRed;
      red->nodeData.red.input1 = static_cast<char *>(scratchbuff) + rxOffset;
      red->nodeData.red.input2 =
          static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset;
      red->nodeData.red.output =
          i == ringSteps - 1
              ? static_cast<char *>(recvbuff) + sliceOffset
              : static_cast<char *>(scratchbuff) + rxOffset;
      red->nodeData.red.count = sliceCount;
      red->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
      red->nodeData.red.datatype = datatype;
      red->nodeData.red.redOp = op;
      red->numParents = 1;
      red->numChildren = i + 1 < ringSteps ? 1 : 0;
      FLAGCXCHECK(allocDagNodeDeps(red));
      FLAGCXCHECK(setDagNodeParent(red, 0, p2pIdx));
      if (red->numChildren != 0) {
        red->children[0] = p2pIndex(s, i + 1);
      }
      runnerState->numPendingNodes++;
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank ReduceScatter topology (%d "
        "slices)",
        nranks, numSlices);
  return validateDagNodes(runnerState);
}

static flagcxResult_t buildUniRunnerStateTreeRed(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    void *scratchbuff, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t op, int root, flagcxComm_t comm) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  if (nranks < 1) {
    return flagcxInvalidArgument;
  } else if (nranks == 1) {
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                               sizeof(struct uniRunnerDagNode)));
      if (runnerState->dagNodes == NULL) {
        return flagcxSystemError;
      }
      runnerState->numDagNodes = 1;
      uniRunnerDagNode *node = &runnerState->dagNodes[0];
      node->nodeIdx = 0;
      node->nodeType = uniRunnerDagNodeTypeCpy;
      node->nodeData.cpy.src = const_cast<void *>(sendbuff);
      node->nodeData.cpy.dst = recvbuff;
      node->nodeData.cpy.count = count;
      node->nodeData.cpy.datatype = datatype;
      node->numParents = 0;
      node->numChildren = 0;
      FLAGCXCHECK(allocDagNodeDeps(node));
      flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      runnerState->numPendingNodes = 0;
    }
    return validateDagNodes(runnerState);
  }

  int algoRank = (rank - root + nranks) % nranks;
  int numSlices = runnerState->uniRunnerNSlices;
  int nTotalSteps = 8 * sizeof(int) - __builtin_clz(nranks - 1);
  int recvNodesPerSlice = algoRank ? __builtin_ctz(algoRank) : nTotalSteps;
  if (algoRank && recvNodesPerSlice &&
      nranks - algoRank <= (1 << (recvNodesPerSlice - 1))) {
    recvNodesPerSlice =
        nranks - algoRank - 1
            ? 8 * sizeof(int) - __builtin_clz(nranks - algoRank - 1)
            : 0;
  }
  bool hasSend = algoRank != 0;
  int nodesPerSlice = 2 * recvNodesPerSlice + (hasSend ? 1 : 0);
  int numNodes = numSlices * nodesPerSlice;
  auto recvIndex = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i;
  };
  auto redIndex = [nodesPerSlice](int s, int i) {
    return s * nodesPerSlice + 2 * i + 1;
  };
  auto sendIndex = [nodesPerSlice, recvNodesPerSlice](int s) {
    return s * nodesPerSlice + 2 * recvNodesPerSlice;
  };

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(
      &runnerState->dagNodes, numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  for (int s = 0; s < numSlices; ++s) {
    size_t baseSliceCount = count / numSlices;
    size_t sliceRemainder = count % numSlices;
    size_t sliceCount =
        baseSliceCount + (s < (int)sliceRemainder ? 1 : 0);
    size_t sliceOffset =
        (s * baseSliceCount + std::min(s, (int)sliceRemainder)) * typeSize;
    size_t rxOffset = count * typeSize + sliceOffset;

    for (int i = 0; i < recvNodesPerSlice; ++i) {
      int recvIdx = recvIndex(s, i);
      uniRunnerDagNode *recv = &runnerState->dagNodes[recvIdx];
      recv->nodeIdx = recvIdx;
      recv->nodeType = uniRunnerDagNodeTypeP2p;
      recv->nodeData.p2p.numOps = 1;
      FLAGCXCHECK(flagcxCalloc(&recv->nodeData.p2p.ops,
                               sizeof(struct uniRunnerP2pOpData)));
      int peer = (rank + (1 << i)) % nranks;
      recv->nodeData.p2p.ops[0] =
          {static_cast<char *>(scratchbuff) + rxOffset, sliceCount, peer,
           datatype, flagcxDevicePrimRecv};
      bool hasCommParent = !(s == 0 && i == 0);
      recv->numParents = (hasCommParent ? 1 : 0) + (i > 0 ? 1 : 0);
      bool hasCommChild = s + 1 < numSlices || i + 1 < recvNodesPerSlice ||
                          (hasSend && i == recvNodesPerSlice - 1);
      recv->numChildren = 1 + (hasCommChild ? 1 : 0);
      FLAGCXCHECK(allocDagNodeDeps(recv));
      int parentSlot = 0;
      if (hasCommParent) {
        FLAGCXCHECK(setDagNodeParent(
            recv, parentSlot++,
            s > 0 ? recvIndex(s - 1, i)
                  : recvIndex(numSlices - 1, i - 1)));
      }
      if (i > 0) {
        FLAGCXCHECK(setDagNodeParent(recv, parentSlot, redIndex(s, i - 1)));
      }
      recv->children[0] = redIndex(s, i);
      if (hasCommChild) {
        recv->children[1] =
            s + 1 < numSlices
                ? recvIndex(s + 1, i)
                : (i + 1 < recvNodesPerSlice ? recvIndex(0, i + 1)
                                             : sendIndex(0));
      }
      if (recv->numParents == 0) {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, recv);
      } else {
        runnerState->numPendingNodes++;
      }

      int redIdx = redIndex(s, i);
      uniRunnerDagNode *red = &runnerState->dagNodes[redIdx];
      red->nodeIdx = redIdx;
      red->nodeType = uniRunnerDagNodeTypeRed;
      red->nodeData.red.input1 = static_cast<char *>(scratchbuff) + rxOffset;
      void *input2Base = i == 0 ? const_cast<void *>(sendbuff) : scratchbuff;
      red->nodeData.red.input2 =
          static_cast<char *>(input2Base) + rxOffset - count * typeSize;
      void *outputBase = i == nTotalSteps - 1 ? recvbuff : scratchbuff;
      red->nodeData.red.output =
          static_cast<char *>(outputBase) + rxOffset - count * typeSize;
      red->nodeData.red.count = sliceCount;
      red->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
      red->nodeData.red.datatype = datatype;
      red->nodeData.red.redOp = op;
      red->numParents = 1;
      red->numChildren = i + 1 < recvNodesPerSlice || hasSend ? 1 : 0;
      FLAGCXCHECK(allocDagNodeDeps(red));
      FLAGCXCHECK(setDagNodeParent(red, 0, recvIdx));
      if (red->numChildren != 0) {
        red->children[0] = i + 1 < recvNodesPerSlice
                               ? recvIndex(s, i + 1)
                               : sendIndex(s);
      }
      runnerState->numPendingNodes++;
    }

    if (hasSend) {
      int sendIdx = sendIndex(s);
      uniRunnerDagNode *send = &runnerState->dagNodes[sendIdx];
      send->nodeIdx = sendIdx;
      send->nodeType = uniRunnerDagNodeTypeP2p;
      send->nodeData.p2p.numOps = 1;
      FLAGCXCHECK(flagcxCalloc(&send->nodeData.p2p.ops,
                               sizeof(struct uniRunnerP2pOpData)));
      int peer = (rank - (1 << __builtin_ctz(algoRank)) + nranks) % nranks;
      void *sendBase = recvNodesPerSlice == 0
                           ? const_cast<void *>(sendbuff)
                           : scratchbuff;
      send->nodeData.p2p.ops[0] =
          {static_cast<char *>(sendBase) + sliceOffset, sliceCount, peer,
           datatype, flagcxDevicePrimSend};
      send->numParents = recvNodesPerSlice == 0 ? (s == 0 ? 0 : 1) : 2;
      send->numChildren = s + 1 < numSlices ? 1 : 0;
      FLAGCXCHECK(allocDagNodeDeps(send));
      if (recvNodesPerSlice == 0) {
        if (s == 0) {
          flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, send);
        } else {
          FLAGCXCHECK(setDagNodeParent(send, 0, sendIndex(s - 1)));
          runnerState->numPendingNodes++;
        }
      } else {
        FLAGCXCHECK(setDagNodeParent(
            send, 0, s > 0 ? sendIndex(s - 1)
                           : recvIndex(numSlices - 1,
                                       recvNodesPerSlice - 1)));
        FLAGCXCHECK(setDagNodeParent(
            send, 1, redIndex(s, recvNodesPerSlice - 1)));
        runnerState->numPendingNodes++;
      }
      if (send->numChildren != 0) {
        send->children[0] = sendIndex(s + 1);
      }
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "DAG scheduler initialized with %d-rank Reduce (root %d) topology "
        "(%d slices)",
        nranks, root, numSlices);
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
      nodeDesc.red.datatype = node->nodeData.red.datatype;
      nodeDesc.red.redOp = node->nodeData.red.redOp;
    } else if (node->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.src, bindings,
                                   &nodeDesc.cpy.src));
      FLAGCXCHECK(captureBufferRef(node->nodeData.cpy.dst, bindings,
                                   &nodeDesc.cpy.dst));
      nodeDesc.cpy.count = node->nodeData.cpy.count;
      nodeDesc.cpy.datatype = node->nodeData.cpy.datatype;
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
      // The reduce kernel uses nthreads as its per-trigger loop stride, so it
      // must match the block size used to launch the persistent RED kernel.
      // Loaded DAG JSON stores only operation DAG parameters; nthreads comes
      // from the runtime launch setting.
      dst->nodeData.red.nthreads = runnerState->uniRunnerNThreads;
      dst->nodeData.red.datatype = src.red.datatype;
      dst->nodeData.red.redOp = src.red.redOp;
    } else if (dst->nodeType == uniRunnerDagNodeTypeCpy) {
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.src, bindings, &dst->nodeData.cpy.src));
      FLAGCXCHECK(
          resolveBufferRef(src.cpy.dst, bindings, &dst->nodeData.cpy.dst));
      dst->nodeData.cpy.count = src.cpy.count;
      dst->nodeData.cpy.datatype = src.cpy.datatype;
    } else {
      return flagcxNotSupported;
    }

    if (dst->numParents == 0) {
      if (dst->nodeType == uniRunnerDagNodeTypeRed) {
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
      uniRunnerDagAlgoLocRed, flagcxCommOpAllReduce, count, datatype, op, -1,
      comm);
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
                               count, datatype, flagcxRedNoOp, -1, comm);
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
      flagcxRedNoOp, -1, comm);
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
      uniRunnerDagAlgoRingAR, flagcxCommOpAllReduce, count, datatype, op, -1,
      comm);
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
  runnerState->redSliceWorkersEnabled = true;

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoSlicedAR, flagcxCommOpAllReduce, count, datatype, op, -1,
      comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  runnerState->uniRunnerNRedSlices =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks);
  runnerState->redSliceWorkersEnabled = true;

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * comm->nranks * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = count * comm->nranks * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingRS, flagcxCommOpReduceScatter, count, datatype, op,
      -1, comm);
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
  runnerState->redSliceWorkersEnabled = true;

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  bindings.inputBytes = count * typeSize;
  bindings.outputBytes = count * typeSize;
  bindings.scratchBytes = 2 * count * typeSize;

  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoTreeRed, flagcxCommOpReduce, count, datatype, op, root,
      comm);
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

    // Launch copy
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        current->nodeData.cpy.dst, current->nodeData.cpy.src,
        current->nodeData.cpy.count *
            getFlagcxDataTypeSize(current->nodeData.cpy.datatype),
        flagcxMemcpyDeviceToDevice, runnerState->cpyStream, NULL));

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
             uniRunnerDagNodeTypeRed) {
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
    uniRunnerRedNodeData *red = &current->nodeData.red;
    if (current->nodeType != uniRunnerDagNodeTypeRed ||
        current->numParents > 1 || red->workerCount < 1 ||
        red->nextWorkerToSubmit > red->workerCount ||
        (red->workerCount > 1 && red->completionCounter == NULL)) {
      return flagcxInvalidArgument;
    }
    uint64_t flagIn =
        current->numParents == 0
            ? 0
            : (uintptr_t)getDagNodeFlag(runnerState, current->parents[0]);
    uint64_t flagOut = (uintptr_t)getDagNodeFlag(runnerState, current->nodeIdx);
    if (red->workerCount == 1) {
      int idx = -1;
      FLAGCXCHECK(enqueue(
          (void *)runnerState->fifo->buffer, (uintptr_t)red->input1,
          (uintptr_t)red->input2, (uintptr_t)red->output, red->count,
          red->nthreads, red->datatype, red->redOp, flagIn, flagOut, &idx));
      if (idx == -1) {
        sched_yield();
        return flagcxSuccess;
      }
      red->triggerIdx = idx;
      red->nextWorkerToSubmit = 1;
    } else {
      while (red->nextWorkerToSubmit < red->workerCount) {
        uint32_t workerId = red->nextWorkerToSubmit;
        int idx = -1;
        FLAGCXCHECK(enqueueReduceWorker(
            (void *)runnerState->fifo->buffer, (uintptr_t)red->input1,
            (uintptr_t)red->input2, (uintptr_t)red->output, red->count,
            red->nthreads, red->datatype, red->redOp, workerId,
            red->workerCount, (uintptr_t)red->completionCounter, flagIn,
            flagOut, &idx));
        if (idx == -1) {
          sched_yield();
          return flagcxSuccess;
        }
        if (red->triggerIdx == -1) {
          red->triggerIdx = idx;
        }
        red->nextWorkerToSubmit++;
      }
    }
    flagcxIntruQueueDequeue(&runnerState->redReadyQueue);
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
  runnerState->redSliceWorkersEnabled = false;

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
  FLAGCXCHECK(prepareRedWorkerRuntimeState(runnerState));

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
  deviceAdaptor->streamSynchronize(runnerState->redStream);
  deviceAdaptor->streamSynchronize(runnerState->cpyStream);
  deviceAdaptor->streamSynchronize(runnerState->commStream);

  return flagcxSuccess;
}

flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState) {
  FLAGCXCHECK(cleanupDagScheduler(runnerState));
  FLAGCXCHECK(destroyRedCompletionCounterPool(runnerState));
  return destroyStreamFlagQueue(runnerState);
}
