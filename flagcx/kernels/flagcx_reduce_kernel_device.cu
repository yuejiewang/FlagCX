#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"

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
  uint64_t flagOut = getFlagOut();
  if (flagOut != 0) {
    flagcxStreamFlagState flagState = loadStreamFlagState(flagOut);
    if (isStreamFlagStatePending(flagState)) {
      DeviceAPI::Atomic::store(reinterpret_cast<uint64_t *>(flagOut),
                               (uint64_t)flagcxStreamFlagDone,
                               flagcxDeviceMemoryOrderRelease);
    }
  }
  // Recycle the FIFO slot only after the output flag is visible as DONE, so a
  // host-side re-enqueue cannot overwrite flagOut before dependent streams
  // observe completion.
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

FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxReduceKernel(uint64_t fst, uint64_t snd, uint64_t out, uint64_t count,
                   uint64_t nthreads, uint64_t datatype, uint64_t redOp) {
  // to be implemented by vendors
  int tid = threadIdx.x;
  float *fstPtr = (float *)fst;
  float *sndPtr = (float *)snd;
  float *outPtr = (float *)out;
  for (int i = tid; i < count; i += nthreads) {
    outPtr[i] = fstPtr[i] + sndPtr[i];
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void runReduceExecutor(void *fifoBuffer) {
  FLAGCX_SHARED uint64_t shm[16];
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

    // RED nodes are submitted from the host before they are executable, so the
    // kernel marks the output flag as pending once it has claimed the FIFO
    // slot.
    if (tid == 0 && shm[FLAG_OUT_IDX] != 0) {
      uint64_t flagOut = shm[FLAG_OUT_IDX];
      flagcxStreamFlagState flagState = loadStreamFlagState(flagOut);
      if (flagState == flagcxStreamFlagIdle) {
        DeviceAPI::Atomic::store(reinterpret_cast<uint64_t *>(flagOut),
                                 (uint64_t)flagcxStreamFlagPend,
                                 flagcxDeviceMemoryOrderRelease);
      }
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
    flagcxReduceKernel(fst, snd, out, count, nthreads, datatype, redop);
    FLAGCX_DEVICE_SYNC_THREADS();
    FLAGCX_DEVICE_THREAD_FENCE();

    // (5) signal completion and recycle the FIFO slot
    if (tid == 0) {
      flagcxReduceTrigger *t =
          (flagcxReduceTrigger *)(vBuf + flagcxFifoIdxData) + slot;
      t->setComplete();
    }
  }
}

struct alignas(16) flagcxIpcVector128 {
  uint64_t x;
  uint64_t y;
};

FLAGCX_DEVICE_INLINE_DECORATOR void
waitIpcParents(const flagcxIpcTrigger &trigger,
               const uint64_t *parentFlags) {
  if (FLAGCX_THREAD_IDX_X != 0)
    return;
  for (uint32_t i = 0; i < trigger.numParentFlags; ++i) {
    uint64_t flagAddr = parentFlags[trigger.parentFlagsOffset + i];
    int iter = 0;
    while (!isStreamFlagStateDone(loadStreamFlagState(flagAddr))) {
      DeviceAPI::Intrin::spinBackoff(iter++);
    }
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void
copyIpcBytes(void *dst, const void *src, size_t bytes) {
  uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);
  size_t vectorBytes = 0;
  if (((dstAddr | srcAddr) & (alignof(flagcxIpcVector128) - 1)) == 0) {
    vectorBytes = bytes & ~(sizeof(flagcxIpcVector128) - 1);
    flagcxIpcVector128 *dstVec =
        reinterpret_cast<flagcxIpcVector128 *>(dst);
    const flagcxIpcVector128 *srcVec =
        reinterpret_cast<const flagcxIpcVector128 *>(src);
    size_t nvec = vectorBytes / sizeof(flagcxIpcVector128);
    for (size_t i = FLAGCX_THREAD_IDX_X; i < nvec;
         i += FLAGCX_BLOCK_DIM_X) {
      dstVec[i] = srcVec[i];
    }
  }

  char *dstBytes = static_cast<char *>(dst);
  const char *srcBytes = static_cast<const char *>(src);
  for (size_t i = vectorBytes + FLAGCX_THREAD_IDX_X; i < bytes;
       i += FLAGCX_BLOCK_DIM_X) {
    dstBytes[i] = srcBytes[i];
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void runIpcExecutor(
    void *fifoBuffer, const flagcxDevMem &inputMem,
    const flagcxDevMem &outputMem, const flagcxDevMem &readyMem,
    const uint64_t *parentFlags) {
  FLAGCX_SHARED int claimedIdx;
  uint64_t *buffer = static_cast<uint64_t *>(fifoBuffer);
  int capacity = static_cast<int>(buffer[flagcxFifoIdxCapacity]);
  int emptyIter = 0;

  while (true) {
    if (FLAGCX_THREAD_IDX_X == 0) {
      int absoluteIdx = -1;
      dequeue(buffer, &absoluteIdx);
      claimedIdx = absoluteIdx;
    }
    FLAGCX_DEVICE_SYNC_THREADS();
    int absoluteIdx = claimedIdx;
    if (absoluteIdx < 0) {
      if (DeviceAPI::Atomic::load(buffer + flagcxFifoIdxTerminate,
                                  flagcxDeviceMemoryOrderAcquire) == 1) {
        break;
      }
      DeviceAPI::Intrin::spinBackoff(emptyIter++);
      continue;
    }

    emptyIter = 0;
    int slot = absoluteIdx % capacity;
    flagcxIpcTrigger *trigger =
        reinterpret_cast<flagcxIpcTrigger *>(buffer + flagcxFifoIdxData) +
        slot;

    if (FLAGCX_THREAD_IDX_X == 0) {
      DeviceAPI::Atomic::store(&trigger->state,
                               static_cast<uint32_t>(
                                   flagcxReduceTriggerInprogress),
                               flagcxDeviceMemoryOrderRelease);
      if (trigger->flagOut != 0) {
        uint64_t *flagOut =
            reinterpret_cast<uint64_t *>(trigger->flagOut);
        if (loadStreamFlagState(trigger->flagOut) == flagcxStreamFlagIdle) {
          DeviceAPI::Atomic::store(flagOut,
                                   static_cast<uint64_t>(flagcxStreamFlagPend),
                                   flagcxDeviceMemoryOrderRelease);
        }
      }
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    waitIpcParents(*trigger, parentFlags);
    FLAGCX_DEVICE_SYNC_THREADS();

    const flagcxDevMem &srcMem =
        trigger->srcBufferType == flagcxIpcBufferInput ? inputMem : outputMem;
    const void *src = flagcxGetLocalPointer(srcMem, trigger->srcOffsetBytes);
    void *remoteDst = flagcxGetIntraPointer(
        outputMem, trigger->dstOffsetBytes,
        static_cast<int>(trigger->peerLocalRank));
    copyIpcBytes(remoteDst, src, static_cast<size_t>(trigger->bytes));
    FLAGCX_DEVICE_SYNC_THREADS();

    // Publish payload stores before the remote ready epoch. System scope is
    // required because the destination belongs to a peer GPU/process.
    if (FLAGCX_THREAD_IDX_X == 0) {
      DeviceAPI::Intrin::threadfenceSystem();
      uint64_t *remoteReady = static_cast<uint64_t *>(flagcxGetIntraPointer(
          readyMem, trigger->readySlot * sizeof(uint64_t),
          static_cast<int>(trigger->peerLocalRank)));
      DeviceAPI::Atomic::store(remoteReady, trigger->epoch,
                               flagcxDeviceMemoryOrderRelease);
    }
    FLAGCX_DEVICE_SYNC_THREADS();

    if (FLAGCX_THREAD_IDX_X == 0) {
      uint64_t *localReady = static_cast<uint64_t *>(flagcxGetLocalPointer(
          readyMem, trigger->readySlot * sizeof(uint64_t)));
      int iter = 0;
      while (DeviceAPI::Atomic::load(localReady,
                                     flagcxDeviceMemoryOrderAcquire) <
             trigger->epoch) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
      if (trigger->flagOut != 0) {
        DeviceAPI::Atomic::store(
            reinterpret_cast<uint64_t *>(trigger->flagOut),
            static_cast<uint64_t>(flagcxStreamFlagDone),
            flagcxDeviceMemoryOrderRelease);
      }
      DeviceAPI::Atomic::store(
          &trigger->state,
          static_cast<uint32_t>(flagcxReduceTriggerAvailable),
          flagcxDeviceMemoryOrderRelease);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
  }
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(
    void *redFifoBuffer, void *ipcFifoBuffer, flagcxDevMem inputMem,
    flagcxDevMem outputMem, flagcxDevMem readyMem,
    const uint64_t *ipcParentFlags, int nRedBlocks, int nIpcBlocks) {
  if (FLAGCX_BLOCK_IDX_X < nRedBlocks) {
    runReduceExecutor(redFifoBuffer);
  } else if (FLAGCX_BLOCK_IDX_X < nRedBlocks + nIpcBlocks) {
    runIpcExecutor(ipcFifoBuffer, inputMem, outputMem, readyMem,
                   ipcParentFlags);
  }
}

void flagcxLaunchCollectiveKernel(
    void *redFifoBuffer, void *ipcFifoBuffer, flagcxDevMem_t inputMem,
    flagcxDevMem_t outputMem, flagcxDevMem_t readyMem,
    const uint64_t *ipcParentFlags, size_t nthreads, size_t nRedBlocks,
    size_t nIpcBlocks, flagcxStream_t stream) {
  flagcxDevMem input;
  flagcxDevMem output;
  flagcxDevMem ready;
  if (inputMem != nullptr)
    input = flagcxDevMem(*inputMem);
  if (outputMem != nullptr)
    output = flagcxDevMem(*outputMem);
  if (readyMem != nullptr)
    ready = flagcxDevMem(*readyMem);
  size_t nblocks = nRedBlocks + nIpcBlocks;
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      redFifoBuffer, ipcFifoBuffer, input, output, ready, ipcParentFlags,
      static_cast<int>(nRedBlocks), static_cast<int>(nIpcBlocks));
}
