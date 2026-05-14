#include "flagcx.h"
#include "flagcx_kernel.h"
#include "device_api/comm_traits.h"
#include "debug.h"

#include <cstdio>

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

FLAGCX_DEVICE_INLINE_DECORATOR uint64_t setReduceTriggerStateValue(
    uint64_t controlWord, flagcxReduceTriggerState state) {
  controlWord &= ~(flagcxTriggerMask(flagcxReduceTriggerBitsState)
                   << flagcxReduceTriggerOffState);
  controlWord |= (state & flagcxTriggerMask(flagcxReduceTriggerBitsState))
                 << flagcxReduceTriggerOffState;
  return controlWord;
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
  // Mark the trigger reusable only after the output flag is visible as DONE,
  // so dependent streams never observe an incomplete RED node as finished.
  uint64_t currVal =
      DeviceAPI::Atomic::load(value + 3, flagcxDeviceMemoryOrderAcquire);
  currVal =
      setReduceTriggerStateValue(currVal, flagcxReduceTriggerAvailable);
  DeviceAPI::Atomic::store(value + 3, currVal, flagcxDeviceMemoryOrderRelease);
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagIn() {
  return value[4];
}
FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxReduceTrigger::getFlagOut() {
  return value[5];
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

static inline void flushCollectiveLaunchLogs() {
  if (flagcxDebugFile != NULL) fflush(flagcxDebugFile);
  fflush(stderr);
}

static inline void *getCollectiveNativeStreamHandle(flagcxStream_t stream) {
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  return stream == NULL
             ? NULL
             : reinterpret_cast<void *>(*(FLAGCX_DEVICE_STREAM_PTR)stream);
#else
  (void)stream;
  return NULL;
#endif
}

FLAGCX_GLOBAL_DECORATOR void flagcxCollectiveKernel(void *fifoBuffer) {
  FLAGCX_SHARED uint64_t shm[16];
  uint64_t *vBuf = (uint64_t *)fifoBuffer;
  uint64_t cap = 0;
  uint64_t slot = 0;
  int tid = FLAGCX_THREAD_IDX_X;
  uint64_t nextSlot = static_cast<uint64_t>(FLAGCX_BLOCK_IDX_X);
  if (tid == 0) {
    shm[flagcxRedFifoIdxCapacity] = DeviceAPI::Atomic::load(
        &vBuf[flagcxRedFifoIdxCapacity], flagcxDeviceMemoryOrderAcquire);
  }
  FLAGCX_DEVICE_SYNC_THREADS();
  cap = shm[flagcxRedFifoIdxCapacity];

  while (true) {
    // (1) dequeue one trigger index from this block's strided RED work lane
    if (tid == 0) {
      uint64_t myIdx = nextSlot;
      nextSlot += static_cast<uint64_t>(FLAGCX_GRID_DIM_X);
      shm[SLOT_IDX] = myIdx >= cap ? cap : myIdx;
      if (myIdx < cap) {
        flagcxReduceTrigger *t =
            (flagcxReduceTrigger *)(vBuf + flagcxRedFifoIdxData) + myIdx;
        uint64_t controlWord = DeviceAPI::Atomic::load(
            t->value + 3, flagcxDeviceMemoryOrderAcquire);
        DeviceAPI::Atomic::store(
            t->value + 3,
            setReduceTriggerStateValue(controlWord,
                                       flagcxReduceTriggerInprogress),
            flagcxDeviceMemoryOrderRelease);
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
    slot = shm[SLOT_IDX];
    if (slot == cap) {
      break;
    }

    // RED nodes are submitted from the host before they are executable, so the
    // kernel marks the output flag as pending once it has claimed the trigger.
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
    int emptyIter = 0;
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

    // (5) signal completion and recycle the trigger slot
    if (tid == 0) {
      flagcxReduceTrigger *t =
          (flagcxReduceTrigger *)(vBuf + flagcxRedFifoIdxData) + slot;
      t->setComplete();
    }
  }
}

void flagcxLaunchCollectiveKernel(void *fifoBuffer, size_t nthreads,
                                  size_t nblocks, flagcxStream_t stream) {
  void *nativeStream = getCollectiveNativeStreamHandle(stream);
  WARN("flagcxLaunchCollectiveKernel: enter fifoBuffer=%p nthreads=%zu "
       "nblocks=%zu stream=%p nativeStream=%p",
       fifoBuffer, nthreads, nblocks, stream, nativeStream);
  flushCollectiveLaunchLogs();
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  cudaError_t preLaunchErr = cudaPeekAtLastError();
  WARN("flagcxLaunchCollectiveKernel: pre-launch cudaPeekAtLastError=%d (%s)",
       static_cast<int>(preLaunchErr), cudaGetErrorString(preLaunchErr));
  flushCollectiveLaunchLogs();
#endif
  flagcxCollectiveKernel<<<nblocks, nthreads, 0,
                           *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(fifoBuffer);
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
  cudaError_t postLaunchErr = cudaPeekAtLastError();
  WARN("flagcxLaunchCollectiveKernel: post-launch fifoBuffer=%p stream=%p "
       "nativeStream=%p cudaPeekAtLastError=%d (%s)",
       fifoBuffer, stream, nativeStream, static_cast<int>(postLaunchErr),
       cudaGetErrorString(postLaunchErr));
#else
  WARN("flagcxLaunchCollectiveKernel: launched fifoBuffer=%p stream=%p "
       "nativeStream=%p",
       fifoBuffer, stream, nativeStream);
#endif
  flushCollectiveLaunchLogs();
}
