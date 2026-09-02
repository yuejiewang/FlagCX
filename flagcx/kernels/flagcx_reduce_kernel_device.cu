#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/comm_traits.h"

#define SLOT_IDX 4
#define FST_IDX 5
#define SND_IDX 6
#define OUT_IDX 7
#define COUNT_IDX 8
#define NTHREADS_IDX 9
#define DATATYPE_IDX 10
#define REDOP_IDX 11
#define FLAG_IN_IDX 12
#define FLAG_OUT_IDX 13
#define WORKER_INFO_IDX 18
#define COMPLETION_COUNTER_IDX 19

FLAGCX_DEVICE_INLINE_DECORATOR flagcxStreamFlagState
loadStreamFlagState(uint64_t flagAddr) {
  return static_cast<flagcxStreamFlagState>(DeviceAPI::Atomic::load(
      reinterpret_cast<uint64_t *>(flagAddr), flagcxDeviceMemoryOrderAcquire));
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
isStreamFlagStatePending(flagcxStreamFlagState state) {
  return state == flagcxStreamFlagIdle || state == flagcxStreamFlagPend;
}

FLAGCX_DEVICE_INLINE_DECORATOR bool
isStreamFlagStateDone(flagcxStreamFlagState state) {
  return state == flagcxStreamFlagDone;
}

FLAGCX_DEVICE_INLINE_DECORATOR void markStreamFlagPending(uint64_t flagAddr) {
  if (flagAddr == 0) {
    return;
  }
  uint64_t expected = flagcxStreamFlagIdle;
  DeviceAPI::Atomic::compareExchange(
      reinterpret_cast<uint64_t *>(flagAddr), expected,
      static_cast<uint64_t>(flagcxStreamFlagPend),
      flagcxDeviceMemoryOrderAcqRel);
}

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput1() {
  return value[0];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getInput2() {
  return value[1];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getOutput() {
  return value[2];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getCount() {
  return value[3] >> flagcxReduceTriggerOffCount &
         flagcxTriggerMask(flagcxReduceTriggerBitsCount);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getNThreads() {
  return value[3] >> flagcxReduceTriggerOffNThreads &
         flagcxTriggerMask(flagcxReduceTriggerBitsNThreads);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getDatatype() {
  return value[3] >> flagcxReduceTriggerOffDatatype &
         flagcxTriggerMask(flagcxReduceTriggerBitsDatatype);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getRedop() {
  return value[3] >> flagcxReduceTriggerOffRedop &
         flagcxTriggerMask(flagcxReduceTriggerBitsRedop);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getState() {
  return value[3] >> flagcxReduceTriggerOffState &
         flagcxTriggerMask(flagcxReduceTriggerBitsState);
}
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceTrigger::setComplete() {
  publishOutputDone();
  recycle();
}
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxReduceTrigger::publishOutputDone() {
  uint64_t flagOut = getFlagOut();
  if (flagOut != 0) {
    flagcxStreamFlagState flagState = loadStreamFlagState(flagOut);
    if (isStreamFlagStatePending(flagState)) {
      DeviceAPI::Atomic::store(reinterpret_cast<uint64_t *>(flagOut),
                               (uint64_t)flagcxStreamFlagDone,
                               flagcxDeviceMemoryOrderRelease);
    }
  }
}
FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceTrigger::recycle() {
  uint64_t currVal =
      DeviceAPI::Atomic::load(value + 3, flagcxDeviceMemoryOrderAcquire);
  currVal &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
               << flagcxReduceTriggerOffState);
  currVal |= (flagcxReduceTriggerAvailable &
              flagcxTriggerMask(flagcxReduceTriggerBitsState))
             << flagcxReduceTriggerOffState;
  DeviceAPI::Atomic::store(value + 3, currVal, flagcxDeviceMemoryOrderRelease);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagIn() {
  return value[4];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagOut() {
  return value[5];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxReduceTrigger::getReduceWorkerId() {
  return value[6] >> flagcxReduceWorkerOffWorkerId &
         flagcxTriggerMask(flagcxReduceWorkerBitsWorkerId);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxReduceTrigger::getReduceWorkerCount() {
  return value[6] >> flagcxReduceWorkerOffWorkerCount &
         flagcxTriggerMask(flagcxReduceWorkerBitsWorkerCount);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
flagcxReduceTrigger::getReduceCompletionCounter() {
  return value[7];
}
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxReduceTrigger::completeReduceWorker() {
  uint64_t workerCount = getReduceWorkerCount();
  uint32_t *counter =
      reinterpret_cast<uint32_t *>(getReduceCompletionCounter());
  uint32_t oldCompleted =
      DeviceAPI::Atomic::fetchAdd<uint32_t, flagcxDeviceScopeDevice>(
          counter, 1u, flagcxDeviceMemoryOrderAcqRel);
  if (static_cast<uint64_t>(oldCompleted) + 1 == workerCount) {
    FLAGCX_DEVICE_THREAD_FENCE();
    publishOutputDone();
  }
  recycle();
}

FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t dequeue(uint64_t *buffer,
                                                      int *idx) {
  while (true) {
    uint64_t oldConsumed = DeviceAPI::Atomic::load(
        buffer + flagcxFifoIdxConsumed, flagcxDeviceMemoryOrderAcquire);
    uint64_t curProduced = DeviceAPI::Atomic::load(
        buffer + flagcxFifoIdxProduced, flagcxDeviceMemoryOrderAcquire);
    if (oldConsumed >= curProduced) {
      // no-op, task dequeued by other consumers
      *idx = -1;
      break;
    }
    // set consumed from `oldConsumed` to `oldConsumed+1`
    uint64_t expected = oldConsumed;
    if (DeviceAPI::Atomic::compareExchange(buffer + flagcxFifoIdxConsumed,
                                           expected, oldConsumed + 1,
                                           flagcxDeviceMemoryOrderAcqRel)) {
      *idx = oldConsumed;
      break;
    }
  }
  return flagcxSuccess;
}

FLAGCX_DEVICE_INLINE_DECORATOR void computeReduceWorkerRange(
    uint64_t fullCount, uint64_t workerId, uint64_t workerCount,
    uint64_t *elementOffset, uint64_t *elementCount) {
  if (workerCount == 0) {
    *elementOffset = 0;
    *elementCount = fullCount;
    return;
  }
  uint64_t base = fullCount / workerCount;
  uint64_t remainder = fullCount % workerCount;
  *elementCount = base + (workerId < remainder ? 1 : 0);
  *elementOffset =
      workerId * base + (workerId < remainder ? workerId : remainder);
}

FLAGCX_DEVICE_INLINE_DECORATOR void flagcxReduceKernel(
    uint64_t fst, uint64_t snd, uint64_t out, uint64_t elementOffset,
    uint64_t elementCount, uint64_t nthreads, uint64_t datatype,
    uint64_t redOp) {
  // to be implemented by vendors
  int tid = threadIdx.x;
  float *fstPtr = (float *)fst;
  float *sndPtr = (float *)snd;
  float *outPtr = (float *)out;
  for (uint64_t localIndex = tid; localIndex < elementCount;
       localIndex += nthreads) {
    uint64_t globalIndex = elementOffset + localIndex;
    outPtr[globalIndex] = fstPtr[globalIndex] + sndPtr[globalIndex];
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer) {
  FLAGCX_SHARED uint64_t shm[20];
  uint64_t *vBuf = (uint64_t *)fifoBuffer;
  int emptyIter = 0; // backoff counter
  int cap = -1;
  int c = -1;
  int p = -1;
  int term = -1;
  int slot = -1;
  int tid = FLAGCX_THREAD_IDX_X;
  if (tid == 0) {
    shm[flagcxFifoIdxCapacity] = vBuf[flagcxFifoIdxCapacity];
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  cap = shm[flagcxFifoIdxCapacity];

  while (true) {
    // (1) dequeue
    if (tid == 0) {
      shm[flagcxFifoIdxConsumed] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxConsumed], flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxProduced] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxProduced], flagcxDeviceMemoryOrderAcquire);
      shm[flagcxFifoIdxTerminate] = DeviceAPI::Atomic::load(
          &vBuf[flagcxFifoIdxTerminate], flagcxDeviceMemoryOrderAcquire);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    c = shm[flagcxFifoIdxConsumed];
    p = shm[flagcxFifoIdxProduced];
    term = shm[flagcxFifoIdxTerminate];

    // (2) backoff if queue empty
    if (c >= p) {
      // check terminate
      if (term == 1)
        break;
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
      continue;
    }

    // (3) dequeue task (lane 0 in a warp)
    if (tid == 0) {
      int myIdx = -1;
      dequeue(vBuf, &myIdx);
      slot = myIdx & (cap - 1);
      shm[SLOT_IDX] = myIdx < 0 ? cap : slot;
      if (myIdx >= 0) {
        flagcxReduceTrigger *t =
            (flagcxReduceTrigger *)(vBuf + flagcxFifoIdxData) + slot;
        shm[FST_IDX] = t->getInput1();
        shm[SND_IDX] = t->getInput2();
        shm[OUT_IDX] = t->getOutput();
        shm[COUNT_IDX] = t->getCount();
        shm[NTHREADS_IDX] = t->getNThreads();
        shm[DATATYPE_IDX] = t->getDatatype();
        shm[REDOP_IDX] = t->getRedop();
        shm[FLAG_IN_IDX] = t->getFlagIn();
        shm[FLAG_OUT_IDX] = t->getFlagOut();
        shm[WORKER_INFO_IDX] =
            t->getReduceWorkerCount() << flagcxReduceWorkerOffWorkerCount |
            t->getReduceWorkerId();
        shm[COMPLETION_COUNTER_IDX] = t->getReduceCompletionCounter();
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    // sync slot to warp
    slot = shm[SLOT_IDX];
    if (slot == cap) {
      if (term == 1)
        break;
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
      continue;
    }

    if (tid == 0) {
      markStreamFlagPending(shm[FLAG_OUT_IDX]);
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    uint64_t flagIn = shm[FLAG_IN_IDX];
    while (flagIn != 0) {
      flagcxStreamFlagState flagState = loadStreamFlagState(flagIn);
      if (isStreamFlagStateDone(flagState)) {
        break;
      }
      if (isStreamFlagStatePending(flagState)) {
        emptyIter++;
        DeviceAPI::Intrin::spinBackoff(emptyIter);
        continue;
      }
      emptyIter++;
      DeviceAPI::Intrin::spinBackoff(emptyIter);
    }

    // (4) perform reduce task
    emptyIter = 0;
    uint64_t fst = shm[FST_IDX];
    uint64_t snd = shm[SND_IDX];
    uint64_t out = shm[OUT_IDX];
    uint64_t count = shm[COUNT_IDX];
    uint64_t nthreads = shm[NTHREADS_IDX];
    uint64_t datatype = shm[DATATYPE_IDX];
    uint64_t redop = shm[REDOP_IDX];
    uint64_t workerInfo = shm[WORKER_INFO_IDX];
    uint64_t workerId = workerInfo & 0xffffffffull;
    uint64_t workerCount = workerInfo >> flagcxReduceWorkerOffWorkerCount;
    uint64_t elementOffset = 0;
    uint64_t elementCount = 0;
    computeReduceWorkerRange(count, workerId, workerCount, &elementOffset,
                             &elementCount);
    flagcxReduceKernel(fst, snd, out, elementOffset, elementCount, nthreads,
                       datatype, redop);
    FLAGCX_DEVICE_SYNC_THREADS();
    FLAGCX_DEVICE_THREAD_FENCE();

    // (5) signal completion and recycle the FIFO slot
    if (tid == 0) {
      flagcxReduceTrigger *t =
          (flagcxReduceTrigger *)(vBuf + flagcxFifoIdxData) + slot;
      if (workerCount == 0) {
        t->setComplete();
      } else {
        t->completeReduceWorker();
      }
    }
  }
}

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream) {
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(fifoBuffer);
}
