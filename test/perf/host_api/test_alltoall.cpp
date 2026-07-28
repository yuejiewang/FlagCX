#include "perf_common.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

static float alltoAllValue(int sourceRank, int destinationRank, size_t offset,
                           uint64_t batch = 0) {
  // Mix every coordinate before reducing to an exactly representable float.
  // The nonlinear mix avoids a fixed-period pattern that could systematically
  // hide a large, aligned displacement in a full-buffer check.
  constexpr uint64_t kExactFloatModulus = 16777213ULL;
  uint64_t value =
      batch * 0x9e3779b97f4a7c15ULL ^
      static_cast<uint64_t>(sourceRank) * 0xbf58476d1ce4e5b9ULL ^
      static_cast<uint64_t>(destinationRank) * 0x94d049bb133111ebULL ^
      static_cast<uint64_t>(offset);
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  value %= kExactFloatModulus;
  return static_cast<float>(value);
}

static bool envEnabled(const char *name) {
  const char *value = getenv(name);
  return value != nullptr && strtol(value, nullptr, 10) != 0;
}

static void initAlltoAllBuffer(PerfContext &ctx, void *deviceBuffer,
                               size_t countPerPeer, uint64_t batch) {
  const size_t totalCount =
      countPerPeer * static_cast<size_t>(ctx.totalProcs);
  const size_t size = totalCount * sizeof(float);
  float *host = static_cast<float *>(ctx.hello);
  for (int destination = 0; destination < ctx.totalProcs; ++destination) {
    for (size_t offset = 0; offset < countPerPeer; ++offset) {
      host[static_cast<size_t>(destination) * countPerPeer + offset] =
          alltoAllValue(ctx.proc, destination, offset, batch);
    }
  }
  flagcxResult_t result = ctx.devHandle->deviceMemcpy(
      deviceBuffer, ctx.hello, size, flagcxMemcpyHostToDevice, NULL);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d AlltoAll probe initialization failed: %d\n",
            ctx.proc, result);
    ctx.accuracyFailed = true;
  }
}

static void checkAlltoAllBuffer(PerfContext &ctx, const void *deviceBuffer,
                                size_t countPerPeer, uint64_t batch,
                                const char *label) {
  const size_t totalCount =
      countPerPeer * static_cast<size_t>(ctx.totalProcs);
  const size_t size = totalCount * sizeof(float);
  flagcxResult_t result = ctx.devHandle->deviceMemcpy(
      ctx.hello, const_cast<void *>(deviceBuffer), size,
      flagcxMemcpyDeviceToHost, ctx.stream);
  if (result == flagcxSuccess) {
    result = ctx.devHandle->streamSynchronize(ctx.stream);
  }
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d %s copy-back failed: %d\n", ctx.proc, label,
            result);
    ctx.accuracyFailed = true;
    return;
  }

  const float *recv = static_cast<const float *>(ctx.hello);
  for (int source = 0; source < ctx.totalProcs; ++source) {
    for (size_t offset = 0; offset < countPerPeer; ++offset) {
      const size_t index = static_cast<size_t>(source) * countPerPeer + offset;
      const float expected = alltoAllValue(source, ctx.proc, offset, batch);
      if (recv[index] != expected) {
        fprintf(stderr,
                "rank %d %s mismatch at offset %zu: expected %f, got %f\n",
                ctx.proc, label, index, expected, recv[index]);
        ctx.accuracyFailed = true;
        return;
      }
    }
  }
}

static void checkAlltoAllResult(PerfContext &ctx, flagcxResult_t result,
                                const char *label) {
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d %s failed with result %d\n", ctx.proc, label,
            result);
    ctx.accuracyFailed = true;
  }
}

static void runSingleShotCorrectnessCases(PerfContext &ctx) {
  const size_t capacityPerPeer =
      ctx.maxBytes /
      (sizeof(float) * static_cast<size_t>(ctx.totalProcs));
  const size_t countPerPeer = std::min<size_t>(capacityPerPeer, 1024);
  if (countPerPeer == 0) {
    return;
  }

  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 1);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "out-of-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.recvbuff, countPerPeer, 1,
                      "out-of-place AlltoAll probe");

  if (!envEnabled("FLAGCX_USE_HETERO_COMM")) {
    return;
  }

  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 2);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.sendbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "in-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 2,
                      "in-place AlltoAll probe");

  // A forced direct transport deliberately rejects capture instead of
  // silently falling back. Group semantics are covered by the default
  // UniRunner regression run with both direct-transport switches disabled.
  const bool forceDirectTransport =
      envEnabled("FLAGCX_UNIRUNNER_USE_HCCSA2A") ||
      envEnabled("FLAGCX_UNIRUNNER_USE_IPCA2A");
  if (!forceDirectTransport) {
    initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 3);
    flagcxResult_t startResult = flagcxGroupStart(ctx.comm);
    flagcxResult_t operationResult = startResult;
    flagcxResult_t endResult = startResult;
    if (startResult == flagcxSuccess) {
      operationResult =
          flagcxAlltoAll(ctx.sendbuff, ctx.sendbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream);
      endResult = flagcxGroupEnd(ctx.comm);
    }
    checkAlltoAllResult(ctx, startResult, "grouped AlltoAll probe start");
    checkAlltoAllResult(ctx, operationResult, "grouped in-place AlltoAll");
    checkAlltoAllResult(ctx, endResult, "grouped AlltoAll probe end");
    if (operationResult == flagcxSuccess && endResult == flagcxSuccess) {
      checkAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 3,
                          "grouped in-place AlltoAll");
    }
  } else if (ctx.proc == 0 && ctx.color == 0) {
    printf("AlltoAll grouped probe skipped: forced direct transport\n");
  }

  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 4);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "post-group out-of-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.recvbuff, countPerPeer, 4,
                      "post-group out-of-place AlltoAll probe");

  // Reuse the same HCCS staging slots with two distinct payload generations
  // across a chunk boundary. A final check of repeated identical data would
  // not detect an implementation that receives the previous generation.
  if (envEnabled("FLAGCX_UNIRUNNER_USE_HCCSA2A")) {
    constexpr size_t kDefaultHccsChunkBytes = 16 * 1024 * 1024;
    size_t chunkBytes = kDefaultHccsChunkBytes;
    const char *configuredChunk =
        getenv("FLAGCX_UNIRUNNER_HCCS_CHUNK_BYTES");
    if (configuredChunk != nullptr) {
      char *end = nullptr;
      const unsigned long long parsed = strtoull(configuredChunk, &end, 10);
      if (end != configuredChunk && *end == '\0' && parsed != 0 &&
          parsed <= static_cast<unsigned long long>(SIZE_MAX)) {
        chunkBytes = static_cast<size_t>(parsed);
      }
    }
    const size_t largeCountPerPeer =
        chunkBytes / sizeof(float) + 1024;
    if (largeCountPerPeer <= capacityPerPeer) {
      for (uint64_t batch : {5ULL, 6ULL}) {
        initAlltoAllBuffer(ctx, ctx.sendbuff, largeCountPerPeer, batch);
        checkAlltoAllResult(
            ctx,
            flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, largeCountPerPeer,
                           flagcxFloat, ctx.comm, ctx.stream),
            "multi-chunk freshness AlltoAll probe");
        checkAlltoAllBuffer(ctx, ctx.recvbuff, largeCountPerPeer, batch,
                            "multi-chunk freshness AlltoAll probe");
      }
    }
  }
}

static void collFn(PerfContext &ctx, size_t count) {
  flagcxResult_t result =
      flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, count / ctx.totalProcs,
                    flagcxFloat, ctx.comm, ctx.stream);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d AlltoAll failed with result %d\n", ctx.proc,
            result);
    ctx.accuracyFailed = true;
  }
}

static double bwFactorFn(int totalProcs) {
  return static_cast<double>(totalProcs - 1) /
         static_cast<double>(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t countPerPeer = count / ctx.totalProcs;
  float *send = static_cast<float *>(ctx.hello);
  memset(ctx.hello, 0, size);
  for (int destination = 0; destination < ctx.totalProcs; ++destination) {
    for (size_t offset = 0; offset < countPerPeer; ++offset) {
      send[static_cast<size_t>(destination) * countPerPeer + offset] =
          alltoAllValue(ctx.proc, destination, offset);
    }
  }
  flagcxResult_t result =
      ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                                  flagcxMemcpyHostToDevice, nullptr);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d timed AlltoAll initialization failed: %d\n",
            ctx.proc, result);
    ctx.accuracyFailed = true;
    return;
  }
  if (ctx.color == 0 && ctx.printBuffer) {
    printf("rank %d sendbuff = ", ctx.proc);
    for (int destination = 0; destination < ctx.totalProcs; ++destination) {
      if (countPerPeer != 0) {
        printf("%f ",
               send[static_cast<size_t>(destination) * countPerPeer]);
      }
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t countPerPeer = count / ctx.totalProcs;
  memset(ctx.hello, 0, size);
  flagcxResult_t result =
      ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                                  flagcxMemcpyDeviceToHost, ctx.stream);
  if (result == flagcxSuccess)
    result = ctx.devHandle->streamSynchronize(ctx.stream);
  if (result != flagcxSuccess) {
    fprintf(stderr, "rank %d timed AlltoAll copy-back failed: %d\n", ctx.proc,
            result);
    ctx.accuracyFailed = true;
    return;
  }

  const float *recv = static_cast<const float *>(ctx.hello);
  int correct = 1;
  bool mismatchReported = false;
  for (int source = 0; source < ctx.totalProcs; ++source) {
    for (size_t offset = 0; offset < countPerPeer; ++offset) {
      const size_t index = static_cast<size_t>(source) * countPerPeer + offset;
      const float expected = alltoAllValue(source, ctx.proc, offset);
      if (recv[index] != expected) {
        if (!mismatchReported) {
          fprintf(stderr,
                  "rank %d wrong output at offset %zu (source rank %d, local "
                  "offset %zu), expected %f, got %f\n",
                  ctx.proc, index, source, offset, expected, recv[index]);
          mismatchReported = true;
        }
        correct = 0;
        ctx.accuracyFailed = true;
      }
    }
  }

  if (ctx.color == 0 && ctx.printBuffer) {
    printf("rank %d recvbuff = ", ctx.proc);
    for (int source = 0; source < ctx.totalProcs; ++source) {
      if (countPerPeer != 0) {
        printf("%f ", recv[static_cast<size_t>(source) * countPerPeer]);
      }
    }
    printf("\n");
    printf("rank %d all-to-all correctness = %d\n", ctx.proc, correct);
  }
}

int main(int argc, char *argv[]) {
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  runSingleShotCorrectnessCases(ctx);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
