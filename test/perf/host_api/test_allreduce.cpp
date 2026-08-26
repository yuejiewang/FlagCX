#include "perf_common.h"

namespace {

constexpr const char *kUseIpcRingAllReduceEnv =
    "FLAGCX_PERF_USE_IPC_RING_AR";

static bool envEnabled(const char *name) {
  const char *value = getenv(name);
  return value != nullptr && atoi(value) == 1;
}

static void configureIpcRingAllReduceForPerf() {
  if (!envEnabled(kUseIpcRingAllReduceEnv)) return;

  // Keep the benchmark on the public host API while forcing its AllReduce
  // dispatch through UniRunner. UNIRUNNER_USE_IPCAR now selects IpcRingAR;
  // setting both variables before perfSetup ensures communicator creation and
  // the first cached parameter lookup observe the requested path.
  setenv("FLAGCX_USE_HETERO_COMM", "1", 1);
  setenv("FLAGCX_UNIRUNNER_USE_IPCAR", "1", 1);
}

} // namespace

static void collFn(PerfContext &ctx, size_t count) {
  flagcxAllReduce(ctx.sendbuff, ctx.recvbuff, count, ctx.datatype, ctx.redOp,
                  ctx.comm, ctx.stream);
}

static double bwFactorFn(int totalProcs) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  double factor = (double)(2 * (totalProcs - 1)) / (double)(totalProcs);
  if (envLocRed != NULL && atoi(envLocRed) == 1) {
    factor = 1;
  } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
    factor = (double)(totalProcs - 1) / (double)(totalProcs);
  }
  return factor;
}

static void dataInitFn(PerfContext &ctx, size_t size, size_t count) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const bool useLocRed = envLocRed != NULL && atoi(envLocRed) == 1;
  perfInitReductionBuffers(ctx, size, count, useLocRed);
}

static void postIterFn(PerfContext &ctx, size_t, size_t count) {
  const char *envLocRed = getenv("FLAGCX_UNIRUNNER_USE_LOCRED");
  const char *envRingAG = getenv("FLAGCX_UNIRUNNER_USE_RINGAG");
  perfPrintBuffer(ctx, count, "recvbuff");

  if (envLocRed != NULL && atoi(envLocRed) == 1) {
    perfCheckLocalReduction(ctx, count, "local-reduce");
  } else if (envRingAG != NULL && atoi(envRingAG) == 1) {
    perfCheckRingGather(ctx, count, "ring-gather");
  } else {
    perfCheckUniformReduction(ctx, count,
                              perfReduceAcrossRanks(ctx.redOp,
                                                    ctx.totalProcs),
                              "all-reduce");
  }
}

int main(int argc, char *argv[]) {
  configureIpcRingAllReduceForPerf();
  PerfContext ctx;
  perfSetup(ctx, argc, argv);
  perfEnableTypedReduction(ctx);
  perfWarmup(ctx, collFn);
  perfBenchmarkLoop(ctx, collFn, bwFactorFn, dataInitFn, postIterFn);
  const int result = ctx.accuracyFailed ? 1 : 0;
  perfTeardown(ctx);
  return result;
}
