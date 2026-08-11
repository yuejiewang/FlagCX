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
#include "device_api/flagcx_device.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <algorithm>
#include <assert.h>
#include <cerrno>
#include <cstdint>
#include <math.h>
#include <limits>
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
FLAGCX_PARAM(UniRunnerNRedBlocks, "UNIRUNNER_NREDBLOCKS", 1);
FLAGCX_PARAM(UniRunnerNIpcBlocks, "UNIRUNNER_NIPCBLOCKS", 1);
FLAGCX_PARAM(UniRunnerIpcChunkSize, "UNIRUNNER_IPCCHUNKSIZE", 262144);
FLAGCX_PARAM(UniRunnerNRedSlices, "UNIRUNNER_NREDSLICES", 0);
FLAGCX_PARAM(UniRunnerRedSliceSize, "UNIRUNNER_REDSLICESIZE", 65536);

static flagcxResult_t loadUniRunnerLaunchConfig(uint64_t *nthreads,
                                                uint64_t *nRedBlocks,
                                                uint64_t *nIpcBlocks) {
  const int64_t rawNthreads = flagcxParamUniRunnerNThreads();
  const int64_t rawNRedBlocks = flagcxParamUniRunnerNRedBlocks();
  const int64_t rawNIpcBlocks = flagcxParamUniRunnerNIpcBlocks();
  const int64_t maxBlocks = std::numeric_limits<int>::max();

  if (rawNthreads <= 0 || rawNRedBlocks < 0 || rawNIpcBlocks < 0) {
    WARN("UniRunner requires NTHREADS > 0, NREDBLOCKS >= 0 and "
         "NIPCBLOCKS >= 0");
    return flagcxInvalidArgument;
  }
  if (rawNRedBlocks > maxBlocks || rawNIpcBlocks > maxBlocks ||
      rawNRedBlocks > maxBlocks - rawNIpcBlocks) {
    WARN("UniRunner executor block count exceeds the supported launch range");
    return flagcxInvalidArgument;
  }

  *nthreads = static_cast<uint64_t>(rawNthreads);
  *nRedBlocks = static_cast<uint64_t>(rawNRedBlocks);
  *nIpcBlocks = static_cast<uint64_t>(rawNIpcBlocks);
  return flagcxSuccess;
}

static flagcxResult_t loadUniRunnerDagBuilderConfig(uint64_t *nSlices,
                                                    uint64_t *nRedSlices,
                                                    uint64_t *redSliceSize) {
  const int64_t rawNSlices = flagcxParamUniRunnerNSlices();
  const int64_t rawNRedSlices = flagcxParamUniRunnerNRedSlices();
  const int64_t rawRedSliceSize = flagcxParamUniRunnerRedSliceSize();
  const int64_t maxBuilderCount = std::numeric_limits<int>::max();
  if (nSlices == NULL || nRedSlices == NULL || redSliceSize == NULL) {
    return flagcxInvalidArgument;
  }
  if (rawNSlices <= 0 || rawNSlices > maxBuilderCount ||
      rawNRedSlices < 0 || rawNRedSlices > maxBuilderCount ||
      rawRedSliceSize < 0) {
    WARN("UniRunner requires 0 < NSLICES <= INT_MAX, "
         "0 <= NREDSLICES <= INT_MAX and REDSLICESIZE >= 0");
    return flagcxInvalidArgument;
  }
  *nSlices = static_cast<uint64_t>(rawNSlices);
  *nRedSlices = static_cast<uint64_t>(rawNRedSlices);
  *redSliceSize = static_cast<uint64_t>(rawRedSliceSize);
  return flagcxSuccess;
}

flagcxResult_t validateUniRunnerLaunchConfig(flagcxCommOp_t commOp) {
  uint64_t nthreads = 0;
  uint64_t nRedBlocks = 0;
  uint64_t nIpcBlocks = 0;
  FLAGCXCHECK(
      loadUniRunnerLaunchConfig(&nthreads, &nRedBlocks, &nIpcBlocks));
  uint64_t nSlices = 0;
  uint64_t nRedSlices = 0;
  uint64_t redSliceSize = 0;
  FLAGCXCHECK(loadUniRunnerDagBuilderConfig(&nSlices, &nRedSlices,
                                            &redSliceSize));

  if (nRedBlocks == 0 &&
      (commOp == flagcxCommOpReduce || commOp == flagcxCommOpAllReduce ||
       commOp == flagcxCommOpReduceScatter)) {
    WARN("FLAGCX_UNIRUNNER_NREDBLOCKS=0 is not valid for %s",
         uniRunnerCommOpToString(commOp));
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static flagcxResult_t ensureUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    const void *sendbuff, void *recvbuff, size_t bytes);
static flagcxResult_t ensureUniRunnerIpcReadyStorage(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    size_t requiredSlots);

static bool isValidUniRunnerDataType(flagcxDataType_t datatype) {
  return static_cast<int>(datatype) >= 0 &&
         static_cast<int>(datatype) < flagcxNumTypes &&
         getFlagcxDataTypeSize(datatype) != 0;
}

static bool isValidUniRunnerRedOp(flagcxRedOp_t op) {
  return static_cast<int>(op) >= static_cast<int>(flagcxSum) &&
         static_cast<int>(op) < static_cast<int>(flagcxNumRedOps);
}

flagcxResult_t validateUniRunnerReduceArgs(size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxRedOp_t op) {
  if (!isValidUniRunnerDataType(datatype) || !isValidUniRunnerRedOp(op)) {
    return flagcxInvalidArgument;
  }
  if (count > std::numeric_limits<size_t>::max() /
                  getFlagcxDataTypeSize(datatype)) {
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

flagcxResult_t checkedUniRunnerTypeBytes(size_t count, size_t multiplier,
                                         flagcxDataType_t datatype,
                                         size_t *bytes) {
  if (bytes == NULL || validateUniRunnerReduceArgs(count, datatype,
                                                    flagcxSum) !=
                          flagcxSuccess) {
    return flagcxInvalidArgument;
  }
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  const size_t maxSize = std::numeric_limits<size_t>::max();
  if (multiplier != 0 && count > maxSize / multiplier) {
    return flagcxInvalidArgument;
  }
  const size_t elements = count * multiplier;
  if (typeSize != 0 && elements > maxSize / typeSize) {
    return flagcxInvalidArgument;
  }
  *bytes = elements * typeSize;
  return flagcxSuccess;
}

flagcxResult_t checkedUniRunnerDagNodeCount(size_t outerCount,
                                            size_t nodesPerOuter,
                                            size_t extraNodes,
                                            int *nodeCount) {
  if (nodeCount == NULL ||
      extraNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  const size_t maxProduct =
      static_cast<size_t>(std::numeric_limits<int>::max()) - extraNodes;
  if (outerCount != 0 && nodesPerOuter > maxProduct / outerCount) {
    return flagcxInvalidArgument;
  }
  *nodeCount = static_cast<int>(outerCount * nodesPerOuter + extraNodes);
  return flagcxSuccess;
}

namespace {

constexpr char kUniRunnerDagCachePathEnv[] = "FLAGCX_UNIRUNNER_DAG_CACHE_PATH";
constexpr uint64_t kUniRunnerDagAlgorithmHashSchemaVersion = 1;
constexpr uint64_t kUniRunnerDagFnvOffsetBasis = 14695981039346656037ull;
constexpr uint64_t kUniRunnerDagFnvPrime = 1099511628211ull;

struct uniRunnerDagRuntimeBindings {
  const void *inputBase = NULL;
  void *outputBase = NULL;
  void *scratchBase = NULL;
  size_t inputBytes = 0;
  size_t outputBytes = 0;
  size_t scratchBytes = 0;
};

static flagcxRedOp_t effectiveUniRunnerRedOp(flagcxRedOp_t requestedOp,
                                             bool isFinalReduction) {
  if (requestedOp == flagcxAvg && !isFinalReduction) {
    return flagcxSum;
  }
  return requestedOp;
}

static void setUniRunnerAvgDivisor(flagcxUniRunnerState *runnerState,
                                   flagcxRedOp_t op, uint64_t divisor) {
  runnerState->avgDivisor = (op == flagcxAvg && divisor != 0) ? divisor : 1;
}

std::mutex gUniRunnerDagCacheMutex;
std::unordered_map<size_t, std::vector<std::shared_ptr<uniRunnerDagTemplate>>>
    gUniRunnerDagCache;
std::unordered_set<std::string> gUniRunnerDagLoadedPaths;

static size_t hashCombine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

static uint64_t hashAlgorithmUint64(uint64_t hashValue, uint64_t value) {
  // Hash fixed-width integers byte-by-byte so algoHash is stable across host
  // word sizes and standard-library implementations.
  for (unsigned int byte = 0; byte < sizeof(value); ++byte) {
    hashValue ^= value & 0xffu;
    hashValue *= kUniRunnerDagFnvPrime;
    value >>= 8;
  }
  return hashValue;
}

static bool bufferMatchesElementOffset(const void *candidate, const void *base,
                                       size_t count, int rank,
                                       size_t typeSize) {
  if (candidate == NULL || base == NULL || rank < 0 || typeSize == 0) {
    return false;
  }
  const size_t rankValue = static_cast<size_t>(rank);
  const size_t maxSize = std::numeric_limits<size_t>::max();
  if (rankValue != 0 && count > maxSize / rankValue) {
    return false;
  }
  const size_t elementOffset = count * rankValue;
  if (elementOffset > maxSize / typeSize) {
    return false;
  }
  const size_t byteOffset = elementOffset * typeSize;
  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
  if (baseAddress > std::numeric_limits<uintptr_t>::max() - byteOffset) {
    return false;
  }
  return reinterpret_cast<uintptr_t>(candidate) == baseAddress + byteOffset;
}

static int positiveModulo(int64_t value, int modulus) {
  assert(modulus > 0);
  const int64_t remainder = value % static_cast<int64_t>(modulus);
  return static_cast<int>(remainder < 0 ? remainder + modulus : remainder);
}

static bool uniRunnerDagCacheKeysEqual(const uniRunnerDagCacheKey &lhs,
                                       const uniRunnerDagCacheKey &rhs) {
  return lhs.algoType == rhs.algoType && lhs.algoHash == rhs.algoHash &&
         lhs.commOp == rhs.commOp &&
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
  if (delta > bytes ||
      delta > static_cast<uintptr_t>(std::numeric_limits<int64_t>::max())) {
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
                 const uniRunnerDagRuntimeBindings &bindings,
                 size_t accessBytes, void **ptr) {
  const char *base = NULL;
  size_t bytes = 0;
  switch (ref.bufferType) {
    case uniRunnerDagBufferTypeNone:
      *ptr = NULL;
      return accessBytes == 0 ? flagcxSuccess : flagcxInvalidArgument;
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
      static_cast<size_t>(ref.offsetBytes) > bytes ||
      accessBytes > bytes - static_cast<size_t>(ref.offsetBytes)) {
    return flagcxInvalidArgument;
  }

  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
  const uintptr_t offset = static_cast<uintptr_t>(ref.offsetBytes);
  if (baseAddress > std::numeric_limits<uintptr_t>::max() - offset) {
    return flagcxInvalidArgument;
  }
  *ptr = reinterpret_cast<void *>(baseAddress + offset);
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

  const size_t maxSize = std::numeric_limits<size_t>::max();
  size_t divisor = static_cast<size_t>(nranks);
  const size_t nSlices = static_cast<size_t>(runnerState->uniRunnerNSlices);
  const size_t redSliceSize =
      static_cast<size_t>(runnerState->uniRunnerRedSliceSize);
  // If the mathematical divisor exceeds SIZE_MAX, it is necessarily larger
  // than count and the positive-count ceil division is exactly one.
  if (divisor > maxSize / nSlices) {
    return 1;
  }
  divisor *= nSlices;
  if (divisor > maxSize / redSliceSize) {
    return 1;
  }
  divisor *= redSliceSize;
  return std::max<size_t>(1, count / divisor + (count % divisor != 0));
}

static flagcxResult_t setEffectiveUniRunnerRedSlices(
    flagcxUniRunnerState *runnerState, size_t count, int nranks) {
  if (runnerState == NULL || nranks <= 0) {
    return flagcxInvalidArgument;
  }
  const size_t resolved =
      resolveEffectiveUniRunnerRedSlices(runnerState, count, nranks);
  if (resolved >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  runnerState->uniRunnerNRedSlices = static_cast<uint64_t>(resolved);
  return flagcxSuccess;
}

static uniRunnerDagCacheKey makeUniRunnerDagCacheKey(
    uniRunnerDagAlgoType algoType,
    const uniRunnerDagAlgorithmConfig &algorithmConfig,
    flagcxCommOp_t commOp, size_t count, flagcxDataType_t datatype,
    flagcxRedOp_t redOp, int root, flagcxComm_t comm) {
  uniRunnerDagCacheKey key{};
  key.algoType = algoType;
  key.algoHash = getUniRunnerDagAlgorithmHash(algoType, algorithmConfig);
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

uint64_t getUniRunnerDagAlgorithmHash(
    uniRunnerDagAlgoType algoType,
    const uniRunnerDagAlgorithmConfig &algorithmConfig) {
  uint64_t hashValue = kUniRunnerDagFnvOffsetBasis;
  hashValue =
      hashAlgorithmUint64(hashValue, kUniRunnerDagAlgorithmHashSchemaVersion);
  hashValue = hashAlgorithmUint64(hashValue, static_cast<uint64_t>(algoType));
  switch (algoType) {
    case uniRunnerDagAlgoLocRed:
    case uniRunnerDagAlgoRingAG:
    case uniRunnerDagAlgoRingAR:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoGroupedAG:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.groupSize);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoSlicedAR:
    case uniRunnerDagAlgoRingRS:
    case uniRunnerDagAlgoTreeRed:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue =
          hashAlgorithmUint64(hashValue, algorithmConfig.numRedSlices);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoIpcAR:
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.numSlices);
      hashValue =
          hashAlgorithmUint64(hashValue, algorithmConfig.numRedSlices);
      hashValue = hashAlgorithmUint64(hashValue, algorithmConfig.topologyHash);
      hashValue = hashAlgorithmUint64(
          hashValue, static_cast<uint64_t>(algorithmConfig.bufferMode));
      break;
    case uniRunnerDagAlgoDummy:
    default:
      break;
  }
  return hashValue;
}

size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key) {
  size_t hashValue = 0;
  // Cache format is serialization metadata rather than a collective semantic
  // key field. Salt the persisted pattern hash so incompatible formats never
  // share a cache filename.
  hashValue = hashCombine(
      hashValue, static_cast<size_t>(kUniRunnerDagCacheFormatVersion));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.algoType));
  hashValue = hashCombine(
      hashValue,
      static_cast<size_t>(static_cast<uint32_t>(key.algoHash & 0xffffffffu)));
  hashValue = hashCombine(
      hashValue, static_cast<size_t>(static_cast<uint32_t>(key.algoHash >> 32)));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.commOp));
  hashValue = hashCombine(hashValue, key.count);
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.datatype));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.redOp));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.rank));
  hashValue = hashCombine(hashValue, static_cast<size_t>(key.nranks));
  hashValue = hashCombine(
      hashValue,
      static_cast<size_t>(static_cast<int64_t>(key.root) + int64_t{1}));
  return hashValue;
}

static flagcxResult_t allocDagNodeDeps(uniRunnerDagNode *node) {
  node->pendingParents = 0;
  if (node->numParents > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->parents, node->numParents));
  }
  if (node->numChildren > 0) {
    FLAGCXCHECK(flagcxCalloc(&node->children, node->numChildren));
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
  if (runnerState == NULL) {
    return flagcxInvalidArgument;
  }
  if (runnerState->numDagNodes < 0) {
    return flagcxInternalError;
  }
  if (runnerState->numDagNodes == 0) {
    return runnerState->dagNodes == NULL ? flagcxSuccess
                                         : flagcxInternalError;
  }
  if (runnerState->dagNodes == NULL) {
    return flagcxInternalError;
  }

  const int numDagNodes = runnerState->numDagNodes;
  uniRunnerDagNode *dagNodes = runnerState->dagNodes;
  size_t numEdges = 0;

  for (int i = 0; i < numDagNodes; i++) {
    uniRunnerDagNode *node = &dagNodes[i];
    if (node->nodeIdx != i) {
      return flagcxInternalError;
    }
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
    switch (node->nodeType) {
      case uniRunnerDagNodeTypeP2p:
        if (node->nodeData.p2p.numOps < 0 ||
            (node->nodeData.p2p.numOps > 0 &&
             node->nodeData.p2p.ops == NULL)) {
          return flagcxInternalError;
        }
        for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; ++opIdx) {
          if (!isValidUniRunnerDataType(
                  node->nodeData.p2p.ops[opIdx].datatype)) {
            return flagcxInvalidArgument;
          }
        }
        break;
      case uniRunnerDagNodeTypeRed:
        if (runnerState->uniRunnerNRedBlocks == 0) {
          WARN("UniRunner DAG contains RED work but NREDBLOCKS is zero");
          return flagcxInvalidArgument;
        }
        if (!isValidUniRunnerDataType(node->nodeData.red.datatype) ||
            !isValidUniRunnerRedOp(node->nodeData.red.redOp) ||
            node->nodeData.red.count >
                flagcxTriggerMask(flagcxReduceTriggerBitsCount) ||
            node->nodeData.red.nthreads == 0 ||
            node->nodeData.red.nthreads >
                flagcxTriggerMask(flagcxReduceTriggerBitsNThreads) ||
            node->nodeData.red.nthreads != runnerState->uniRunnerNThreads) {
          return flagcxInvalidArgument;
        }
        break;
      case uniRunnerDagNodeTypeCpy:
        if (!isValidUniRunnerDataType(node->nodeData.cpy.datatype)) {
          return flagcxInvalidArgument;
        }
        break;
      case uniRunnerDagNodeTypeIpc:
        if (runnerState->uniRunnerNIpcBlocks == 0) {
          WARN("UniRunner DAG contains IPC work but NIPCBLOCKS is zero");
          return flagcxInvalidArgument;
        }
        break;
      default:
        return flagcxInternalError;
    }
    if (static_cast<size_t>(node->numParents) >
        std::numeric_limits<size_t>::max() - numEdges) {
      return flagcxInvalidArgument;
    }
    numEdges += static_cast<size_t>(node->numParents);
  }

  try {
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
        uint64_t edge =
            (static_cast<uint64_t>(static_cast<uint32_t>(i)) << 32) |
            static_cast<uint32_t>(childIdx);
        std::unordered_set<uint64_t>::iterator it = dagEdges.find(edge);
        if (it == dagEdges.end()) {
          return flagcxInternalError;
        }
        dagEdges.erase(it);
      }
    }

    return dagEdges.empty() ? flagcxSuccess : flagcxInternalError;
  } catch (...) {
    return flagcxSystemError;
  }
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
  if (newCapacity >
      std::numeric_limits<size_t>::max() / sizeof(*runnerState->streamFlags)) {
    return flagcxInvalidArgument;
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
    const size_t maxSize = std::numeric_limits<size_t>::max();
    const size_t maxCapacity =
        std::min(maxSize / sizeof(*runnerState->streamFlags),
                 maxSize / sizeof(uint64_t));
    if (requiredFlags > maxCapacity) {
      return flagcxInvalidArgument;
    }
    size_t newCapacity = runnerState->streamFlagsCapacity == 0
                             ? 1
                             : runnerState->streamFlagsCapacity;
    while (newCapacity < requiredFlags) {
      if (newCapacity > maxCapacity / 2) {
        newCapacity = requiredFlags;
        break;
      }
      newCapacity *= 2;
    }

    void *newPool = NULL;
    flagcxResult_t res = deviceAdaptor->deviceMalloc(
        &newPool, newCapacity * sizeof(uint64_t), flagcxMemDevice, NULL);
    if (res != flagcxSuccess) {
      return res;
    }
    res = resizeStreamFlagAddressQueue(runnerState, newCapacity);
    if (res != flagcxSuccess) {
      (void)deviceAdaptor->deviceFree(newPool, flagcxMemDevice, NULL);
      return res;
    }
    if (runnerState->streamFlagsPool != NULL) {
      res = deviceAdaptor->deviceFree(runnerState->streamFlagsPool,
                                      flagcxMemDevice, NULL);
      if (res != flagcxSuccess) {
        (void)deviceAdaptor->deviceFree(newPool, flagcxMemDevice, NULL);
        return res;
      }
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
    case uniRunnerDagNodeTypeIpc:
      return runnerState->redStream;
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
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
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
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
        effectiveUniRunnerRedOp(op, true);

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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(1, nGroups, 1, &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  size_t localBaseOffset =
      static_cast<size_t>(groupIdx) * groupChunkCount * typeSize;
  for (int step = 0; step < nGroups; step++) {
    int nodeIdx = step;
    bool isLastStep = (step == nGroups - 1);
    int numIntraPeers = groupSize - 1;
    int numOps = 0;
    FLAGCXCHECK(checkedUniRunnerDagNodeCount(
        2, numIntraPeers, isLastStep ? 0 : 2, &numOps));

    runnerState->dagNodes[nodeIdx].nodeIdx = nodeIdx;
    runnerState->dagNodes[nodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
    runnerState->dagNodes[nodeIdx].nodeData.p2p.numOps = numOps;
    if (numOps > 0) {
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[nodeIdx].nodeData.p2p.ops, numOps));
    }

    for (int i = 0; i < numIntraPeers; i++) {
      int locSendPeer = positiveModulo(
          static_cast<int64_t>(locRank) + i + 1, groupSize);
      int locRecvPeer = positiveModulo(
          static_cast<int64_t>(locRank) - i - 1, groupSize);

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
      size_t sendGroupIdx = positiveModulo(
          static_cast<int64_t>(groupIdx) + step + 1, nGroups);
      size_t recvGroupIdx = positiveModulo(
          static_cast<int64_t>(groupIdx) - step - 1, nGroups);
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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 1,
                                           &runnerState->numDagNodes));
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  int nodesPerSlice = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(3, nranks - 1, 0, &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, numNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
          effectiveUniRunnerRedOp(op, i == nranks - 2);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i + 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);

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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateSlicedAR called, count=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration - handle uneven distribution
  size_t baseRankChunkCount = count / nranks;
  size_t rankChunkRemainder = count % nranks;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce * numRedSlices) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  int nodesPerStep = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(1, numRedSlices, 2, &nodesPerStep));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(nodesPerStep, nranks - 1, 0,
                                           &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);

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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(op, i == nranks - 2);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i + 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i, nranks);

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

static bool getIpcBufferOffset(const void *ptr, const void *base, size_t bytes,
                               size_t *offset) {
  if (ptr == NULL || base == NULL)
    return false;
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t b = reinterpret_cast<uintptr_t>(base);
  // One-past-end is valid for a zero-count slice; the trigger will carry
  // bytes==0 and only exchange its ready epoch.
  if (p < b || p - b > bytes)
    return false;
  *offset = static_cast<size_t>(p - b);
  return true;
}

static int globalRankToLocalRank(flagcxComm_t comm, int globalRank) {
  for (int localRank = 0; localRank < comm->localRanks; ++localRank) {
    if (comm->localRankToRank[localRank] == globalRank)
      return localRank;
  }
  return -1;
}

static flagcxResult_t buildUniRunnerStateIpcAR(
    flagcxUniRunnerState *runnerState, const void *sendbuff, void *recvbuff,
    size_t count, flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm) {
  if (comm->localRanks != comm->nranks)
    return flagcxNotSupported;
  if (comm->nranks <= 1 || count == 0) {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  }

  FLAGCXCHECK(buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                          count, datatype, op, comm));

  size_t totalBytes = 0;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 1, datatype, &totalBytes));
  uint32_t readySlot = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeP2p)
      continue;

    const uniRunnerP2pOpData *sendOp = NULL;
    const uniRunnerP2pOpData *recvOp = NULL;
    for (int opIdx = 0; opIdx < node->nodeData.p2p.numOps; ++opIdx) {
      const uniRunnerP2pOpData *candidate = &node->nodeData.p2p.ops[opIdx];
      if (candidate->type == flagcxDevicePrimSend)
        sendOp = candidate;
      else if (candidate->type == flagcxDevicePrimRecv)
        recvOp = candidate;
    }
    if (sendOp == NULL || recvOp == NULL) {
      return flagcxNotSupported;
    }

    size_t srcOffset = 0;
    flagcxIpcBufferType srcBufferType = flagcxIpcBufferInput;
    // Prefer output for aliased buffers: both handles describe the same local
    // storage and output is the destination surface carried around the ring.
    if (getIpcBufferOffset(sendOp->addr, recvbuff, totalBytes, &srcOffset)) {
      srcBufferType = flagcxIpcBufferOutput;
    } else if (getIpcBufferOffset(sendOp->addr, sendbuff, totalBytes,
                                  &srcOffset)) {
      srcBufferType = flagcxIpcBufferInput;
    } else {
      return flagcxInvalidArgument;
    }

    size_t localRecvOffset = 0;
    if (!getIpcBufferOffset(recvOp->addr, recvbuff, totalBytes,
                            &localRecvOffset)) {
      return flagcxInvalidArgument;
    }
    (void)localRecvOffset;
    int peerLocalRank = globalRankToLocalRank(comm, sendOp->peerRank);
    if (peerLocalRank < 0)
      return flagcxNotSupported;

    uniRunnerP2pOpData *oldOps = node->nodeData.p2p.ops;
    uniRunnerIpcNodeData ipc = {};
    ipc.srcOffsetBytes = srcOffset;
    // Ring chunks retain their global output-buffer offset at every rank. The
    // local recv op refers to the chunk arriving from prevRank, while this
    // send op writes its own source chunk to nextRank at srcOffset.
    ipc.dstOffsetBytes = srcOffset;
    FLAGCXCHECK(checkedUniRunnerTypeBytes(sendOp->count, 1, sendOp->datatype,
                                          &ipc.bytes));
    ipc.srcBufferType = srcBufferType;
    ipc.peerLocalRank = peerLocalRank;
    if (readySlot == std::numeric_limits<uint32_t>::max()) {
      return flagcxInvalidArgument;
    }
    ipc.readySlot = readySlot++;
    ipc.parentFlagsOffset = 0;
    ipc.triggerIdx = -1;
    free(oldOps);
    node->nodeType = uniRunnerDagNodeTypeIpc;
    node->nodeData.ipc = ipc;
  }

  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
  runnerState->numPendingNodes = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    node->pendingParents = node->numParents;
    if (node->numParents == 0) {
      if (node->nodeType == uniRunnerDagNodeTypeRed) {
        flagcxIntruQueueEnqueue(&runnerState->redReadyQueue, node);
      } else if (node->nodeType == uniRunnerDagNodeTypeIpc) {
        flagcxIntruQueueEnqueue(&runnerState->ipcReadyQueue, node);
      } else {
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue, node);
      }
    } else {
      runnerState->numPendingNodes++;
    }
  }

  runnerState->ipcReadySlots = static_cast<size_t>(readySlot);
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
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
    TRACE(FLAGCX_UNIRUNNER, "uniRunnerNRedSlices auto set to %lu",
          runnerState->uniRunnerNRedSlices);
  }
  int numSlices = runnerState->uniRunnerNSlices;
  int numRedSlices = runnerState->uniRunnerNRedSlices;

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d initUniRunnerStateRingRS called, recvcount=%lu, numSlices=%d, "
        "numRedSlices=%d",
        comm->rank, count, numSlices, numRedSlices);

  int nextRank = positiveModulo(static_cast<int64_t>(rank) + 1, nranks);
  int prevRank = positiveModulo(static_cast<int64_t>(rank) - 1, nranks);
  size_t typeSize = getFlagcxDataTypeSize(datatype);
  size_t baseRankChunkCount = count;

  // Nodes per slice chain:
  // (P2P + Reduce * numRedSlices) * (nranks - 1)
  int nodesPerStep = 0;
  FLAGCXCHECK(
      checkedUniRunnerDagNodeCount(1, numRedSlices, 1, &nodesPerStep));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(nodesPerStep, nranks - 1, 0,
                                           &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops, 2));

      int txChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 1, nranks);
      int rxChunk =
          positiveModulo(static_cast<int64_t>(rank) - i - 2, nranks);

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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(op, i == nranks - 2);

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

  if (nranks < 1 || rank < 0 || rank >= nranks || root < 0 ||
      root >= nranks) {
    return flagcxInvalidArgument;
  }
  int algoRank = positiveModulo(static_cast<int64_t>(rank) - root,
                                nranks); // Rotate ranks so root is 0
  if (nranks == 1) {
    // For single rank, do local cpy if out-of-place, otherwise no-op
    if (count > 0 && sendbuff != recvbuff) {
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes, 1));
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
    FLAGCXCHECK(
        setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
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
  int redNodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      recvNodesPerSlice, numRedSlices, 0, &redNodesPerSlice));
  int nodesPerSlice = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(
      1, redNodesPerSlice,
      static_cast<size_t>(sendNodesPerSlice) + recvNodesPerSlice,
      &nodesPerSlice));
  int numNodes = 0;
  FLAGCXCHECK(checkedUniRunnerDagNodeCount(numSlices, nodesPerSlice, 0,
                                           &numNodes));

  TRACE(FLAGCX_UNIRUNNER,
        "rank %d (algoRank %d) initUniRunnerStateTreeReduce called, count=%lu, "
        "numSlices=%d, numRedSlices=%d, recvSteps %d, sendSteps %d",
        comm->rank, algoRank, count, numSlices, numRedSlices, recvNodesPerSlice,
        sendNodesPerSlice);

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[recvNodeIdx].nodeData.p2p.ops,
          runnerState->dagNodes[recvNodeIdx].nodeData.p2p.numOps));

      // Recv Node
      int peer = positiveModulo(
          static_cast<int64_t>(rank) + (int64_t{1} << i), nranks);
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
        runnerState->dagNodes[redNodeIdx].nodeData.red.redOp =
            effectiveUniRunnerRedOp(
                op, algoRank == 0 && i == nTotalSteps - 1);

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
      FLAGCXCHECK(flagcxCalloc(
          &runnerState->dagNodes[sendNodeIdx].nodeData.p2p.ops,
          runnerState->dagNodes[sendNodeIdx].nodeData.p2p.numOps));

      int peer = positiveModulo(
          static_cast<int64_t>(rank) -
              (int64_t{1} << __builtin_ctz(algoRank)),
          nranks);

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
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
  runnerState->numPendingNodes = 0;
}

static flagcxResult_t
prepareUniRunnerDagExecutionPlan(flagcxUniRunnerState *runnerState) {
  if (runnerState == NULL || runnerState->numDagNodes < 0) {
    return flagcxInvalidArgument;
  }
  return compileUniRunnerDagExecutionPlan(
      runnerState->dagNodes, static_cast<size_t>(runnerState->numDagNodes),
      &runnerState->dagPlan);
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
  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());
  if (dagTemplate.nodes.size() > maxInt) {
    return flagcxInvalidArgument;
  }
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    if (node.parents.size() > maxInt || node.children.size() > maxInt ||
        node.p2pOps.size() > maxInt) {
      return flagcxInvalidArgument;
    }
  }

  resetDagSchedulerRuntimeState(runnerState);
  runnerState->numDagNodes = static_cast<int>(dagTemplate.nodes.size());
  if (runnerState->numDagNodes == 0) {
    runnerState->dagNodes = NULL;
    return flagcxSuccess;
  }

  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           runnerState->numDagNodes));
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
        FLAGCXCHECK(
            flagcxCalloc(&dst->nodeData.p2p.ops, dst->nodeData.p2p.numOps));
      }
      for (int opIdx = 0; opIdx < dst->nodeData.p2p.numOps; opIdx++) {
        const uniRunnerDagP2pOpDesc &srcOp = src.p2pOps[opIdx];
        uniRunnerP2pOpData *dstOp = &dst->nodeData.p2p.ops[opIdx];
        dstOp->count = srcOp.count;
        dstOp->peerRank = srcOp.peerRank;
        dstOp->datatype = srcOp.datatype;
        dstOp->type = srcOp.type;
        size_t opBytes = 0;
        FLAGCXCHECK(checkedUniRunnerTypeBytes(srcOp.count, 1, srcOp.datatype,
                                              &opBytes));
        FLAGCXCHECK(
            resolveBufferRef(srcOp.buffer, bindings, opBytes, &dstOp->addr));
      }
    } else if (dst->nodeType == uniRunnerDagNodeTypeRed) {
      dst->nodeData.red.triggerIdx = -1;
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(src.red.count, 1,
                                            src.red.datatype, &opBytes));
      FLAGCXCHECK(resolveBufferRef(src.red.input1, bindings, opBytes,
                                   &dst->nodeData.red.input1));
      FLAGCXCHECK(resolveBufferRef(src.red.input2, bindings, opBytes,
                                   &dst->nodeData.red.input2));
      FLAGCXCHECK(resolveBufferRef(src.red.output, bindings, opBytes,
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
      size_t opBytes = 0;
      FLAGCXCHECK(checkedUniRunnerTypeBytes(src.cpy.count, 1,
                                            src.cpy.datatype, &opBytes));
      FLAGCXCHECK(resolveBufferRef(src.cpy.src, bindings, opBytes,
                                   &dst->nodeData.cpy.src));
      FLAGCXCHECK(resolveBufferRef(src.cpy.dst, bindings, opBytes,
                                   &dst->nodeData.cpy.dst));
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
            "uniRunner DAG cache miss, algo_name=%s algo_hash=%llu "
            "commOp=%s pattern_hash=%lu",
            uniRunnerDagAlgoTypeToString(key.algoType),
            static_cast<unsigned long long>(key.algoHash),
            uniRunnerCommOpToString(key.commOp), hashValue);
      return flagcxSuccess;
    }
  }

  TRACE(FLAGCX_UNIRUNNER,
        "uniRunner DAG cache hit, algo_name=%s algo_hash=%llu commOp=%s "
        "pattern_hash=%lu",
        uniRunnerDagAlgoTypeToString(key.algoType),
        static_cast<unsigned long long>(key.algoHash),
        uniRunnerCommOpToString(key.commOp), hashValue);
  FLAGCXCHECK(
      materializeUniRunnerDagTemplate(runnerState, *dagTemplate, bindings));
  FLAGCXCHECK(validateDagNodes(runnerState));
  FLAGCXCHECK(prepareUniRunnerDagExecutionPlan(runnerState));
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
  destroyUniRunnerDagExecutionPlan(&runnerState->dagPlan);
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
  FLAGCXCHECK(prepareUniRunnerDagExecutionPlan(runnerState));
  return cacheBuiltUniRunnerDag(runnerState, key, bindings);
}

flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, 2);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoLocRed, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
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
  runnerState->avgDivisor = 1;
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      count, static_cast<size_t>(comm->nranks), datatype,
      &bindings.outputBytes));

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.groupSize = static_cast<uint64_t>(groupSize);
  algorithmConfig.bufferMode =
      bufferMatchesElementOffset(sendbuff, recvbuff, count, comm->rank,
                                 typeSize)
          ? uniRunnerDagBufferModeInPlace
          : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key =
      makeUniRunnerDagCacheKey(uniRunnerDagAlgoGroupedAG, algorithmConfig,
                               flagcxCommOpAllGather, count, datatype,
                               flagcxRedNoOp, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateGroupedAG(runnerState, sendbuff, recvbuff, count,
                                        datatype, comm, groupSize);
  });
}

flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  runnerState->avgDivisor = 1;
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAG, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, flagcxRedNoOp, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateRingAG(runnerState, sendbuff, recvbuff, count,
                                     datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingAR, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
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
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoSlicedAR, algorithmConfig, flagcxCommOpAllReduce, count,
      datatype, op, -1, comm);
  return initUniRunnerStateCached(runnerState, key, bindings, [&]() {
    return buildUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm);
  });
}

flagcxResult_t initUniRunnerStateIpcAR(flagcxUniRunnerState *runnerState,
                                       const void *sendbuff, void *recvbuff,
                                       size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  if (runnerState->uniRunnerIpcChunkSize < 16 ||
      runnerState->uniRunnerIpcChunkSize % 16 != 0) {
    WARN("UniRunner IPCCHUNKSIZE must be a positive multiple of 16 bytes");
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));
  size_t bytes = 0;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(count, 1, datatype, &bytes));
  // IPC nodes carry runtime Window/DevMem bindings and therefore intentionally
  // bypass the static JSON DAG cache. The underlying SlicedAR topology remains
  // unchanged and all existing cached algorithms are preserved.
  FLAGCXCHECK(buildUniRunnerStateIpcAR(runnerState, sendbuff, recvbuff, count,
                                       datatype, op, comm));
  FLAGCXCHECK(prepareUniRunnerDagExecutionPlan(runnerState));
  if (runnerState->ipcReadySlots == 0)
    return flagcxSuccess;
  FLAGCXCHECK(ensureUniRunnerIpcDataViews(runnerState, comm, sendbuff,
                                          recvbuff, bytes));
  return ensureUniRunnerIpcReadyStorage(runnerState, comm,
                                        runnerState->ipcReadySlots);
}

flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm) {
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  FLAGCXCHECK(checkedUniRunnerTypeBytes(
      count, static_cast<size_t>(comm->nranks), datatype,
      &bindings.inputBytes));
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.outputBytes));
  bindings.scratchBytes = bindings.inputBytes;

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      bufferMatchesElementOffset(recvbuff, sendbuff, count, comm->rank,
                                 typeSize)
          ? uniRunnerDagBufferModeInPlace
          : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoRingRS, algorithmConfig, flagcxCommOpReduceScatter,
      count, datatype, op, -1, comm);
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
  setUniRunnerAvgDivisor(runnerState, op, comm->nranks);
  FLAGCXCHECK(
      setEffectiveUniRunnerRedSlices(runnerState, count, comm->nranks));

  uniRunnerDagRuntimeBindings bindings;
  bindings.inputBase = sendbuff;
  bindings.outputBase = recvbuff;
  bindings.scratchBase = scratchbuff;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 1, datatype, &bindings.inputBytes));
  bindings.outputBytes = bindings.inputBytes;
  FLAGCXCHECK(
      checkedUniRunnerTypeBytes(count, 2, datatype, &bindings.scratchBytes));

  uniRunnerDagAlgorithmConfig algorithmConfig{};
  algorithmConfig.numSlices = runnerState->uniRunnerNSlices;
  algorithmConfig.numRedSlices = runnerState->uniRunnerNRedSlices;
  algorithmConfig.bufferMode =
      sendbuff == recvbuff ? uniRunnerDagBufferModeInPlace
                           : uniRunnerDagBufferModeOutOfPlace;
  uniRunnerDagCacheKey key = makeUniRunnerDagCacheKey(
      uniRunnerDagAlgoTreeRed, algorithmConfig, flagcxCommOpReduce, count,
      datatype, op, root, comm);
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
    size_t copyBytes = 0;
    FLAGCXCHECK(checkedUniRunnerTypeBytes(
        current->nodeData.cpy.count, 1, current->nodeData.cpy.datatype,
        &copyBytes));
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        current->nodeData.cpy.dst, current->nodeData.cpy.src,
        copyBytes,
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
  } else if (runnerState->dagNodes[nodeIdx].nodeType ==
             uniRunnerDagNodeTypeIpc) {
    flagcxIntruQueueEnqueue(&runnerState->ipcReadyQueue,
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

static flagcxResult_t allocUniRunnerIpcReadyBuffer(void **ptr, size_t bytes) {
  return deviceAdaptor->deviceMalloc(ptr, bytes, flagcxMemDevice, NULL);
}

static flagcxResult_t freeUniRunnerIpcReadyBuffer(void *ptr) {
  if (ptr == NULL)
    return flagcxSuccess;
  return deviceAdaptor->deviceFree(ptr, flagcxMemDevice, NULL);
}

static flagcxResult_t destroyUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState) {
  if (runnerState->ipcOwner == NULL)
    return flagcxSuccess;
  flagcxComm_t comm = runnerState->ipcOwner;
  bool aliased = runnerState->ipcInputMem != NULL &&
                 runnerState->ipcInputMem == runnerState->ipcOutputMem;
  if (runnerState->ipcInputMem != NULL && !aliased) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->ipcInputMem));
  }
  if (runnerState->ipcOutputMem != NULL) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->ipcOutputMem));
  }
  runnerState->ipcInputMem = NULL;
  runnerState->ipcOutputMem = NULL;
  runnerState->ipcInputBase = NULL;
  runnerState->ipcOutputBase = NULL;
  runnerState->ipcDataBytes = 0;
  return flagcxSuccess;
}

static flagcxResult_t ensureUniRunnerIpcDataViews(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    const void *sendbuff, void *recvbuff, size_t bytes) {
  if (runnerState->ipcInputMem != NULL && runnerState->ipcOutputMem != NULL &&
      runnerState->ipcInputBase == sendbuff &&
      runnerState->ipcOutputBase == recvbuff) {
    runnerState->ipcDataBytes = std::max(runnerState->ipcDataBytes, bytes);
    return flagcxSuccess;
  }

  FLAGCXCHECK(destroyUniRunnerIpcDataViews(runnerState));
  runnerState->ipcOwner = comm;
  flagcxResult_t res = flagcxSuccess;

  FLAGCXCHECKGOTO(flagcxDevMemCreate(comm, recvbuff, bytes, NULL,
                                     &runnerState->ipcOutputMem),
                  res, fail);
  if (sendbuff == recvbuff) {
    runnerState->ipcInputMem = runnerState->ipcOutputMem;
  } else {
    FLAGCXCHECKGOTO(flagcxDevMemCreate(
                        comm, const_cast<void *>(sendbuff), bytes, NULL,
                        &runnerState->ipcInputMem),
                    res, fail);
  }
  if (!runnerState->ipcInputMem->hasWindow ||
      !runnerState->ipcOutputMem->hasWindow) {
    res = flagcxNotSupported;
    goto fail;
  }
  runnerState->ipcInputBase = sendbuff;
  runnerState->ipcOutputBase = recvbuff;
  runnerState->ipcDataBytes = bytes;
  return flagcxSuccess;

fail:
  (void)destroyUniRunnerIpcDataViews(runnerState);
  return res;
}

static flagcxResult_t ensureUniRunnerIpcReadyStorage(
    flagcxUniRunnerState *runnerState, flagcxComm_t comm,
    size_t requiredSlots) {
  if (requiredSlots == 0)
    return flagcxSuccess;
  if (requiredSlots >
      std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return flagcxInvalidArgument;
  }
  if (runnerState->ipcReadyMem != NULL &&
      runnerState->ipcReadyCapacity >= requiredSlots) {
    return flagcxSuccess;
  }

  if (runnerState->ipcReadyMem != NULL) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->ipcReadyMem));
    runnerState->ipcReadyMem = NULL;
  }
  if (runnerState->ipcReadyBuffer != NULL) {
    FLAGCXCHECK(
        freeUniRunnerIpcReadyBuffer(runnerState->ipcReadyBuffer));
    runnerState->ipcReadyBuffer = NULL;
  }
  runnerState->ipcReadyCapacity = 0;

  size_t bytes = requiredSlots * sizeof(uint64_t);
  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECKGOTO(allocUniRunnerIpcReadyBuffer(
                      &runnerState->ipcReadyBuffer, bytes),
                  res, fail);
  FLAGCXCHECKGOTO(deviceAdaptor->deviceMemset(runnerState->ipcReadyBuffer, 0,
                                               bytes, flagcxMemDevice, NULL),
                  res, fail);
  FLAGCXCHECKGOTO(flagcxDevMemCreate(
                      comm, runnerState->ipcReadyBuffer, bytes, NULL,
                      &runnerState->ipcReadyMem),
                  res, fail);
  if (!runnerState->ipcReadyMem->hasWindow) {
    res = flagcxNotSupported;
    goto fail;
  }
  runnerState->ipcReadyCapacity = requiredSlots;
  return flagcxSuccess;

fail:
  if (runnerState->ipcReadyMem != NULL) {
    (void)flagcxDevMemDestroy(comm, runnerState->ipcReadyMem);
    runnerState->ipcReadyMem = NULL;
  }
  if (runnerState->ipcReadyBuffer != NULL) {
    (void)freeUniRunnerIpcReadyBuffer(runnerState->ipcReadyBuffer);
    runnerState->ipcReadyBuffer = NULL;
  }
  runnerState->ipcReadyCapacity = 0;
  return res;
}

static flagcxResult_t prepareUniRunnerIpcParentFlags(
    flagcxUniRunnerState *runnerState) {
  if (runnerState->ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(
        runnerState->ipcParentFlagsDevice, flagcxMemDevice, NULL));
    runnerState->ipcParentFlagsDevice = NULL;
    runnerState->ipcParentFlagsCount = 0;
  }

  size_t numParentFlags = 0;
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    const uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeIpc)
      continue;
    if (node->numParents < 0 ||
        static_cast<size_t>(node->numParents) >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()) -
                numParentFlags) {
      return flagcxInvalidArgument;
    }
    numParentFlags += static_cast<size_t>(node->numParents);
  }
  if (numParentFlags >
      std::numeric_limits<size_t>::max() / sizeof(uint64_t)) {
    return flagcxInvalidArgument;
  }

  std::vector<uint64_t> parentFlags;
  try {
    parentFlags.reserve(numParentFlags);
  } catch (...) {
    return flagcxSystemError;
  }
  for (int nodeIdx = 0; nodeIdx < runnerState->numDagNodes; ++nodeIdx) {
    uniRunnerDagNode *node = &runnerState->dagNodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeIpc)
      continue;
    node->nodeData.ipc.parentFlagsOffset =
        static_cast<uint32_t>(parentFlags.size());
    for (int parentIdx = 0; parentIdx < node->numParents; ++parentIdx) {
      parentFlags.push_back(reinterpret_cast<uint64_t>(
          getDagNodeFlag(runnerState, node->parents[parentIdx])));
    }
  }
  if (parentFlags.empty())
    return flagcxSuccess;

  size_t bytes = parentFlags.size() * sizeof(uint64_t);
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      reinterpret_cast<void **>(&runnerState->ipcParentFlagsDevice), bytes,
      flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      runnerState->ipcParentFlagsDevice, parentFlags.data(), bytes,
      flagcxMemcpyHostToDevice, NULL, NULL));
  runnerState->ipcParentFlagsCount = parentFlags.size();
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
    // The current algorithms only create single-parent RED nodes. Multi-parent
    // dependencies are handled for P2P/CPY nodes by emitting one stream wait
    // per parent; RED nodes would need an explicit fan-in flag if that ever
    // changes.
    if (current->numParents > 1) {
      return flagcxInvalidArgument;
    }
    int idx = -1;
    FLAGCXCHECK(enqueue(
        (void *)runnerState->fifo->buffer,
        (uintptr_t)current->nodeData.red.input1,
        (uintptr_t)current->nodeData.red.input2,
        (uintptr_t)current->nodeData.red.output, current->nodeData.red.count,
        current->nodeData.red.nthreads, current->nodeData.red.datatype,
        current->nodeData.red.redOp, flagIn, flagOut, &idx));
    if (idx == -1) {
      sched_yield();
      break; // FIFO full, skip for now
    }
    // Dequeue
    flagcxIntruQueueDequeue(&runnerState->redReadyQueue);
    current->nodeData.red.triggerIdx = idx;
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  while (!flagcxIntruQueueEmpty(&runnerState->ipcReadyQueue)) {
    uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->ipcReadyQueue);
    uint64_t flagOut =
        reinterpret_cast<uint64_t>(getDagNodeFlag(runnerState,
                                                  current->nodeIdx));
    int idx = -1;
    FLAGCXCHECK(enqueueIpc(
        runnerState->ipcFifo->buffer, current->nodeData.ipc.srcOffsetBytes,
        current->nodeData.ipc.dstOffsetBytes, current->nodeData.ipc.bytes,
        runnerState->uniRunnerIpcChunkSize,
        current->nodeData.ipc.srcBufferType,
        current->nodeData.ipc.peerLocalRank, current->nodeData.ipc.readySlot,
        runnerState->ipcEpoch, current->nodeData.ipc.parentFlagsOffset,
        static_cast<uint32_t>(current->numParents), flagOut, &idx));
    if (idx == -1) {
      sched_yield();
      break;
    }
    flagcxIntruQueueDequeue(&runnerState->ipcReadyQueue);
    current->nodeData.ipc.triggerIdx = idx;
    FLAGCXCHECK(notifyChildrenScheduled(runnerState, current));
  }

  return flagcxSuccess;
}

flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  runnerState->dagNodes = NULL;
  runnerState->numDagNodes = 0;
  runnerState->dagPlan.topoOrder = NULL;
  runnerState->dagPlan.numNodes = 0;
  runnerState->dagPlan.numHostNodes = 0;
  runnerState->dagPlan.numRedNodes = 0;
  runnerState->dagPlan.numIpcNodes = 0;
  runnerState->streamFlagsSize = 0;
  runnerState->ipcReadySlots = 0;

  FLAGCXCHECK(loadUniRunnerDagBuilderConfig(
      &runnerState->uniRunnerNSlices, &runnerState->uniRunnerNRedSlices,
      &runnerState->uniRunnerRedSliceSize));
  FLAGCXCHECK(loadUniRunnerLaunchConfig(
      &runnerState->uniRunnerNThreads, &runnerState->uniRunnerNRedBlocks,
      &runnerState->uniRunnerNIpcBlocks));
  runnerState->uniRunnerIpcChunkSize = flagcxParamUniRunnerIpcChunkSize();
  runnerState->avgDivisor = 1;
  // Set device context
  FLAGCXCHECK(deviceAdaptor->setDevice(hcomm->cudaDev));

  // Create FIFO
  runnerState->fifo = new flagcxFifo();
  FLAGCXCHECK(runnerState->fifo->flagcxRedFifoInit());
  // hcomm->proxyState->uniRunnerState.fifo->buffer is the host pointer
  // hcomm->uniRunnerFifoBuffer stores the device pointer to fifo buffer
  FLAGCXCHECK(deviceAdaptor->hostGetDevicePointer(
      &hcomm->uniRunnerFifoBuffer, (void *)runnerState->fifo->buffer));

  runnerState->ipcFifo = new flagcxFifo();
  FLAGCXCHECK(runnerState->ipcFifo->flagcxIpcFifoInit());
  FLAGCXCHECK(deviceAdaptor->hostGetDevicePointer(
      &runnerState->ipcFifoDevicePtr,
      static_cast<void *>(runnerState->ipcFifo->buffer)));

  // Initialize queues
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->ipcReadyQueue);
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
  FLAGCXCHECK(
      hcomm->proxyState->uniRunnerState.ipcFifo->flagcxIpcFifoDestroy());
  delete hcomm->proxyState->uniRunnerState.ipcFifo;
  hcomm->proxyState->uniRunnerState.ipcFifo = NULL;
  hcomm->proxyState->uniRunnerState.ipcFifoDevicePtr = NULL;

  if (hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(
        hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice,
        flagcxMemDevice, NULL));
    hcomm->proxyState->uniRunnerState.ipcParentFlagsDevice = NULL;
    hcomm->proxyState->uniRunnerState.ipcParentFlagsCount = 0;
  }

  return flagcxSuccess;
}

flagcxResult_t runUniRunner(flagcxComm_t comm) {
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxFifo_t fifo = hcomm->proxyState->uniRunnerState.fifo;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  TRACE(FLAGCX_UNIRUNNER, "runUniRunner called");
  FLAGCXCHECK(validateDagNodes(runnerState));
  const size_t numDagNodes = static_cast<size_t>(runnerState->numDagNodes);
  if (runnerState->dagPlan.numNodes != numDagNodes ||
      (numDagNodes != 0 && runnerState->dagPlan.topoOrder == NULL) ||
      runnerState->dagPlan.numHostNodes + runnerState->dagPlan.numRedNodes +
              runnerState->dagPlan.numIpcNodes !=
          numDagNodes) {
    return flagcxInternalError;
  }
  FLAGCXCHECK(prepareDagStreamFlags(runnerState));
  FLAGCXCHECK(prepareUniRunnerIpcParentFlags(runnerState));
  if (runnerState->ipcReadySlots != 0) {
    runnerState->ipcEpoch++;
    if (runnerState->ipcEpoch == 0)
      runnerState->ipcEpoch++;
  }

#ifdef COMPILE_KERNEL_HOST
  if (runnerState->uniRunnerNRedBlocks != 0 ||
      runnerState->uniRunnerNIpcBlocks != 0) {
    FLAGCXCHECK(flagcxLaunchCollectiveKernel(
        hcomm->uniRunnerFifoBuffer, runnerState->ipcFifoDevicePtr,
        runnerState->ipcInputMem, runnerState->ipcOutputMem,
        runnerState->ipcReadyMem, runnerState->ipcParentFlagsDevice,
        runnerState->uniRunnerNThreads, runnerState->uniRunnerNRedBlocks,
        runnerState->uniRunnerNIpcBlocks, runnerState->avgDivisor,
        runnerState->redStream));
  } else {
    TRACE(FLAGCX_UNIRUNNER,
          "runUniRunner: skipping collective kernel launch because both "
          "executor block counts are zero");
  }
#endif

  // Main scheduling loop using DAG-based queue scheduling
  while (true) {
    if (flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->redReadyQueue) &&
        flagcxIntruQueueEmpty(&runnerState->ipcReadyQueue) &&
        runnerState->numPendingNodes == 0) {
      TRACE(
          FLAGCX_UNIRUNNER,
          "runUniRunner: all submitted work drained, terminating runner loop");
      __atomic_store_n(fifo->buffer + flagcxFifoIdxTerminate, 1,
                       __ATOMIC_RELEASE);
      __atomic_store_n(runnerState->ipcFifo->buffer + flagcxFifoIdxTerminate,
                       1, __ATOMIC_RELEASE);
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
  FLAGCXCHECK(destroyStreamFlagQueue(runnerState));
  if (runnerState == NULL)
    return flagcxSuccess;

  flagcxComm_t comm = runnerState->ipcOwner;
  FLAGCXCHECK(destroyUniRunnerIpcDataViews(runnerState));
  if (comm != NULL && runnerState->ipcReadyMem != NULL) {
    FLAGCXCHECK(flagcxDevMemDestroy(comm, runnerState->ipcReadyMem));
  }
  runnerState->ipcReadyMem = NULL;
  if (runnerState->ipcReadyBuffer != NULL) {
    FLAGCXCHECK(
        freeUniRunnerIpcReadyBuffer(runnerState->ipcReadyBuffer));
  }
  runnerState->ipcReadyBuffer = NULL;
  runnerState->ipcReadyCapacity = 0;
  runnerState->ipcReadySlots = 0;
  runnerState->ipcOwner = NULL;
  if (runnerState->ipcParentFlagsDevice != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(runnerState->ipcParentFlagsDevice,
                                          flagcxMemDevice, NULL));
  }
  runnerState->ipcParentFlagsDevice = NULL;
  runnerState->ipcParentFlagsCount = 0;
  return flagcxSuccess;
}
