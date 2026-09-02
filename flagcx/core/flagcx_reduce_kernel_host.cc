#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_PARAM(ReduceFifoCapacity, "REDUCE_FIFO_CAPACITY", FLAGCX_FIFO_CAPACITY);

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setValue(
    uint64_t fst, uint64_t snd, uint64_t out, size_t count, size_t nthreads,
    flagcxDataType_t datatype, flagcxRedOp_t redOp,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
  tmp[0] = fst;
  tmp[1] = snd;
  tmp[2] = out;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[3] = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
               << flagcxReduceTriggerOffCount |
           (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
               << flagcxReduceTriggerOffNThreads |
           (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
               << flagcxReduceTriggerOffDatatype |
           (redOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
               << flagcxReduceTriggerOffRedop |
           (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState |
           (flagcxComputeTriggerReduce &
            flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
               << flagcxReduceTriggerOffComputeType;
  memcpy(this->value, tmp, sizeof(tmp));
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setGemmValue(
    uint64_t a, uint64_t b, uint64_t c, uint32_t m, uint32_t n, uint32_t k,
    uint32_t lda, uint32_t ldb, uint32_t ldc, size_t nthreads,
    flagcxDataType_t datatype, int accumulate,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
  tmp[0] = a;
  tmp[1] = b;
  tmp[2] = c;
  tmp[3] =
      (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
          << flagcxReduceTriggerOffNThreads |
      (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
          << flagcxReduceTriggerOffDatatype |
      (flagcxRedNoOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
          << flagcxReduceTriggerOffRedop |
      (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
          << flagcxReduceTriggerOffState |
      (flagcxComputeTriggerGemm &
       flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
          << flagcxReduceTriggerOffComputeType;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[6] = static_cast<uint64_t>(m) << 32 | n;
  tmp[7] = static_cast<uint64_t>(k) << 32 | lda;
  tmp[8] = static_cast<uint64_t>(ldb) << 32 | ldc;
  tmp[9] = accumulate != 0 ? 1 : 0;
  memcpy(this->value, tmp, sizeof(tmp));
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setReduceWorkerValue(
    uint64_t fst, uint64_t snd, uint64_t out, size_t count, size_t nthreads,
    flagcxDataType_t datatype, flagcxRedOp_t redOp, uint32_t workerId,
    uint32_t workerCount, uint64_t completionCounter,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[12] = {};
  tmp[0] = fst;
  tmp[1] = snd;
  tmp[2] = out;
  tmp[3] = (count & flagcxTriggerMask(flagcxReduceTriggerBitsCount))
               << flagcxReduceTriggerOffCount |
           (nthreads & flagcxTriggerMask(flagcxReduceTriggerBitsNThreads))
               << flagcxReduceTriggerOffNThreads |
           (datatype & flagcxTriggerMask(flagcxReduceTriggerBitsDatatype))
               << flagcxReduceTriggerOffDatatype |
           (redOp & flagcxTriggerMask(flagcxReduceTriggerBitsRedop))
               << flagcxReduceTriggerOffRedop |
           (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
               << flagcxReduceTriggerOffState |
           (flagcxComputeTriggerReduce &
            flagcxTriggerMask(flagcxReduceTriggerBitsComputeType))
               << flagcxReduceTriggerOffComputeType;
  tmp[4] = flagIn;
  tmp[5] = flagOut;
  tmp[6] = static_cast<uint64_t>(workerCount) << 32 | workerId;
  tmp[7] = completionCounter;
  memcpy(this->value, tmp, sizeof(tmp));
}

FLAGCX_HOST_DECORATOR uint64_t flagcxReduceTrigger::pollState() {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  return currVal >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setState(int state) {
  uint64_t currVal = __atomic_load_n(&this->value[3], __ATOMIC_ACQUIRE);
  currVal &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
               << flagcxReduceTriggerOffState);
  currVal |= (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
             << flagcxReduceTriggerOffState;
  __atomic_store_n(&this->value[3], currVal, __ATOMIC_RELEASE);
  TRACE(FLAGCX_KERNEL, "setState called, new state=%llu",
        currVal >> flagcxReduceTriggerOffState &
            flagcxTriggerMask(flagcxReduceTriggerBitsState));
}

FLAGCX_HOST_DECORATOR flagcxResult_t
enqueue(void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
        size_t count, size_t nthreads, flagcxDataType_t datatype,
        flagcxRedOp_t redop, uint64_t flagIn, uint64_t flagOut, int *ret) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  int distance = buffer[flagcxFifoIdxProduced] - buffer[flagcxFifoIdxConsumed];
  // red buffer full, wait for kernel to consume
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  idx = buffer[flagcxFifoIdxProduced] % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;

  // kernel reduce work in progress
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setValue(addr1, addr2, addr3, count, nthreads, datatype, redop,
                    flagcxReduceTriggerEnqueued, flagIn, flagOut);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue red: count=%lu, nthreads=%lu, datatype=%d, redop=%d, idx=%d",
        count, nthreads, datatype, redop, idx);

  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t enqueueReduceWorker(
    void *fifoBuffer, uint64_t addr1, uint64_t addr2, uint64_t addr3,
    size_t count, size_t nthreads, flagcxDataType_t datatype,
    flagcxRedOp_t redop, uint32_t workerId, uint32_t workerCount,
    uint64_t completionCounter, uint64_t flagIn, uint64_t flagOut, int *ret) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  int distance = buffer[flagcxFifoIdxProduced] - buffer[flagcxFifoIdxConsumed];
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  int idx = buffer[flagcxFifoIdxProduced] % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setReduceWorkerValue(
      addr1, addr2, addr3, count, nthreads, datatype, redop, workerId,
      workerCount, completionCounter, flagcxReduceTriggerEnqueued, flagIn,
      flagOut);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue red worker: count=%lu, worker=%u/%u, idx=%d", count,
        workerId, workerCount, idx);
  return flagcxSuccess;
}

FLAGCX_HOST_DECORATOR flagcxResult_t enqueueGemm(
    void *fifoBuffer, uint64_t a, uint64_t b, uint64_t c, uint32_t m,
    uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc,
    size_t nthreads, flagcxDataType_t datatype, int accumulate,
    uint64_t flagIn, uint64_t flagOut, int *ret) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[flagcxFifoIdxCapacity];
  int distance = buffer[flagcxFifoIdxProduced] - buffer[flagcxFifoIdxConsumed];
  if (distance >= capacity) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  int idx = buffer[flagcxFifoIdxProduced] % capacity;
  flagcxReduceTrigger *trigger =
      ((flagcxReduceTrigger *)(buffer + flagcxFifoIdxData)) + idx;
  if (trigger->pollState() != flagcxReduceTriggerAvailable) {
    *ret = -1;
    sched_yield();
    return flagcxSuccess;
  }
  trigger->setGemmValue(a, b, c, m, n, k, lda, ldb, ldc, nthreads, datatype,
                        accumulate, flagcxReduceTriggerEnqueued, flagIn,
                        flagOut);
  __atomic_fetch_add(buffer + flagcxFifoIdxProduced, 1ul, __ATOMIC_RELEASE);
  *ret = idx;
  TRACE(FLAGCX_KERNEL,
        "enqueue gemm: m=%u, n=%u, k=%u, nthreads=%lu, idx=%d", m, n, k,
        nthreads, idx);
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoInit() {
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  uint64_t flagcxReduceFifoCapacity = flagcxParamReduceFifoCapacity();
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer,
                                          flagcxFifoIdxData * sizeof(uint64_t) +
                                              flagcxReduceFifoCapacity *
                                                  sizeof(flagcxReduceTrigger),
                                          flagcxMemHost, NULL));
  buffer[flagcxFifoIdxCapacity] = flagcxReduceFifoCapacity;
  buffer[flagcxFifoIdxConsumed] = 0;
  buffer[flagcxFifoIdxProduced] = 0;
  buffer[flagcxFifoIdxTerminate] = 0;
  memset((void *)(buffer + flagcxFifoIdxData), 0,
         flagcxReduceFifoCapacity * sizeof(flagcxReduceTrigger));
  __sync_synchronize();
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemHost, NULL));
  return flagcxSuccess;
}
