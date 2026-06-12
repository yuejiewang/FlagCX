#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/flagcx_device.h"
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
#define TYPE_IDX 14

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
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getType() {
  return value[3] >> flagcxReduceTriggerOffType &
         flagcxTriggerMask(flagcxReduceTriggerBitsType);
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
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getDevApiOpsPtr() {
  return value[0];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getDevApiNumOps() {
  return value[1];
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

FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCopyBytes(uint8_t *dst, const uint8_t *src, size_t bytes) {
  for (size_t idx = FLAGCX_THREAD_IDX_X; idx < bytes;
       idx += FLAGCX_BLOCK_DIM_X) {
    dst[idx] = src[idx];
  }
}

FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevApiBarrierSync(const flagcxDevApiRuntime &runtime,
                        flagcxCoopBlock coop, uint32_t ctaIndex) {
  if (runtime.intraSize <= 1) {
    coop.sync();
    return;
  }
  if (runtime.barrierPeers == nullptr || runtime.epochBuffer == nullptr ||
      runtime.nBarriers <= 0) {
    coop.sync();
    return;
  }

  uint64_t epoch = DeviceAPI::Atomic::load(
      &runtime.epochBuffer[ctaIndex], flagcxDeviceMemoryOrderRelaxed);
  coop.sync();

  for (int i = coop.threadRank(); i < runtime.intraSize - 1; i += coop.size()) {
    int peer = 1 + runtime.intraRank + i;
    if (peer >= runtime.intraSize)
      peer -= runtime.intraSize;
    uint64_t *slot =
        &runtime.barrierPeers[peer][runtime.intraRank * runtime.nBarriers +
                                    static_cast<int>(ctaIndex)];
    DeviceAPI::Atomic::store(slot, epoch + 1, flagcxDeviceMemoryOrderRelease);
  }

  for (int i = coop.threadRank(); i < runtime.intraSize - 1; i += coop.size()) {
    int peer = 1 + runtime.intraRank + i;
    if (peer >= runtime.intraSize)
      peer -= runtime.intraSize;
    uint64_t *slot =
        &runtime.barrierPeers[runtime.intraRank][peer * runtime.nBarriers +
                                                 static_cast<int>(ctaIndex)];
    int iter = 0;
    while (DeviceAPI::Atomic::load(slot, flagcxDeviceMemoryOrderAcquire) <
           epoch + 1) {
      DeviceAPI::Intrin::spinBackoff(iter++);
    }
  }

  DeviceAPI::Atomic::store(&runtime.epochBuffer[ctaIndex], epoch + 1,
                           flagcxDeviceMemoryOrderRelaxed);
  coop.sync();
}

FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevApiKernel(uint64_t opsPtr, uint64_t numOps, const void *runtimePtr) {
  if (opsPtr == 0 || numOps == 0 || runtimePtr == nullptr) {
    return;
  }

  const flagcxDevApiRuntime *runtime =
      reinterpret_cast<const flagcxDevApiRuntime *>(runtimePtr);
  flagcxCoopBlock coop;
  const flagcxDevApiKernelOp *ops =
      reinterpret_cast<const flagcxDevApiKernelOp *>(opsPtr);

  flagcxDevApiBarrierSync(*runtime, coop, FLAGCX_BLOCK_IDX_X);
  for (uint64_t i = 0; i < numOps; i++) {
    const flagcxDevApiKernelOp &op = ops[i];
    if (op.srcPtr == 0 || op.dstPtr == 0 || op.count == 0 || op.nbytes == 0) {
      continue;
    }
    const uint8_t *src = reinterpret_cast<const uint8_t *>(op.srcPtr);
    uint8_t *dst = reinterpret_cast<uint8_t *>(op.dstPtr);
    if (src == nullptr || dst == nullptr) {
      continue;
    }
    flagcxCopyBytes(dst, src, op.nbytes);
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  flagcxDevApiBarrierSync(*runtime, coop, FLAGCX_BLOCK_IDX_X);
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer,
                                                    const void *devApiRuntimePtr) {
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
        shm[TYPE_IDX] = t->getType();
        if (shm[TYPE_IDX] == flagcxCollectiveTriggerTypeDevApi) {
          shm[FST_IDX] = t->getDevApiOpsPtr();
          shm[COUNT_IDX] = t->getDevApiNumOps();
        }
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
    uint64_t type = shm[TYPE_IDX];
    if (type == flagcxCollectiveTriggerTypeDevApi) {
      flagcxDevApiKernel(fst, count, devApiRuntimePtr);
    } else {
      flagcxReduceKernel(fst, snd, out, count, nthreads, datatype, redop);
    }
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

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream,
                                  const void *devApiRuntimePtr) {
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      fifoBuffer, devApiRuntimePtr);
}
