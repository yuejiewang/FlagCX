#include "perf_common.h"
#include <algorithm>

static float alltoAllValue(int sourceRank, int destinationRank,
                           size_t offset, uint64_t batch = 0) {
  // Keep the generated integer exactly representable as float while making
  // every field of an AlltoAll transfer contribute to the payload.
  constexpr uint64_t kExactFloatModulus = 16777213ULL;
  const uint64_t value =
      (batch * 65537ULL + static_cast<uint64_t>(sourceRank) * 1000003ULL +
       static_cast<uint64_t>(destinationRank) * 1009ULL +
       static_cast<uint64_t>(offset)) %
      kExactFloatModulus;
  return static_cast<float>(value);
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
      const float expected =
          alltoAllValue(source, ctx.proc, offset, batch);
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

  // Out-of-place, then aliased, then out-of-place again exercises one cached
  // DAG with different runtime buffer relationships.
  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 1);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "out-of-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.recvbuff, countPerPeer, 1,
                      "out-of-place AlltoAll probe");

  const char *useHeteroComm = getenv("FLAGCX_USE_HETERO_COMM");
  if (useHeteroComm == nullptr || strtol(useHeteroComm, nullptr, 10) == 0) {
    return;
  }

  // These cases specifically cover the UniRunner path implemented here. The
  // generic timed case above remains valid for every runner backend.
  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 2);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.sendbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "in-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 2,
                      "in-place AlltoAll probe");

  // The aliased operation keeps its snapshot alive until GroupEnd has
  // submitted the whole exchange.
  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 3);
  flagcxResult_t startResult = flagcxGroupStart(ctx.comm);
  flagcxResult_t operationResult = startResult;
  flagcxResult_t endResult = startResult;
  if (startResult == flagcxSuccess) {
    operationResult = flagcxAlltoAll(ctx.sendbuff, ctx.sendbuff, countPerPeer,
                                     flagcxFloat, ctx.comm, ctx.stream);
    // Always close a successfully opened group, including inner error paths.
    endResult = flagcxGroupEnd(ctx.comm);
  }
  checkAlltoAllResult(ctx, startResult, "grouped AlltoAll probe start");
  checkAlltoAllResult(ctx, operationResult, "grouped in-place AlltoAll");
  checkAlltoAllResult(ctx, endResult, "grouped AlltoAll probe end");
  if (operationResult == flagcxSuccess && endResult == flagcxSuccess) {
    checkAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 3,
                        "grouped in-place AlltoAll");
  }

  initAlltoAllBuffer(ctx, ctx.sendbuff, countPerPeer, 4);
  checkAlltoAllResult(
      ctx, flagcxAlltoAll(ctx.sendbuff, ctx.recvbuff, countPerPeer, flagcxFloat,
                         ctx.comm, ctx.stream),
      "post-group out-of-place AlltoAll probe");
  checkAlltoAllBuffer(ctx, ctx.recvbuff, countPerPeer, 4,
                      "post-group out-of-place AlltoAll probe");
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
  return (double)(totalProcs - 1) / (double)(totalProcs);
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t countPerPeer = count / ctx.totalProcs;
  float *send = static_cast<float *>(ctx.hello);
  memset(ctx.hello, 0, size);
  for (int destination = 0; destination < ctx.totalProcs; destination++) {
    for (size_t offset = 0; offset < countPerPeer; offset++) {
      send[static_cast<size_t>(destination) * countPerPeer + offset] =
          alltoAllValue(ctx.proc, destination, offset);
    }
  }
  ctx.devHandle->deviceMemcpy(ctx.sendbuff, ctx.hello, size,
                              flagcxMemcpyHostToDevice, NULL);
  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("sendbuff = ");
    for (int destination = 0; destination < ctx.totalProcs; destination++) {
      if (countPerPeer != 0) {
        printf("%f ", send[static_cast<size_t>(destination) * countPerPeer]);
      }
    }
    printf("\n");
  }
}

static void postIterFn(PerfContext &ctx, size_t size, size_t count) {
  const size_t countPerPeer = count / ctx.totalProcs;
  memset(ctx.hello, 0, size);
  ctx.devHandle->deviceMemcpy(ctx.hello, ctx.recvbuff, size,
                              flagcxMemcpyDeviceToHost, ctx.stream);
  ctx.devHandle->streamSynchronize(ctx.stream);

  const float *recv = static_cast<const float *>(ctx.hello);
  bool mismatchReported = false;
  for (int source = 0; source < ctx.totalProcs; source++) {
    for (size_t offset = 0; offset < countPerPeer; offset++) {
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
        ctx.accuracyFailed = true;
      }
    }
  }

  if ((ctx.proc == 0 || ctx.proc == ctx.totalProcs - 1) && ctx.color == 0 &&
      ctx.printBuffer) {
    printf("recvbuff = ");
    for (int source = 0; source < ctx.totalProcs; source++) {
      if (countPerPeer != 0) {
        printf("%f ", recv[static_cast<size_t>(source) * countPerPeer]);
      }
    }
    printf("\n");
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
