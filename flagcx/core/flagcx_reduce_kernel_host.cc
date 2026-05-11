#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_HOST_DECORATOR void flagcxReduceTrigger::setValue(
    uint64_t fst, uint64_t snd, uint64_t out, size_t count, size_t nthreads,
    flagcxDataType_t datatype, flagcxRedOp_t redOp,
    flagcxReduceTriggerState state, uint64_t flagIn, uint64_t flagOut) {
  uint64_t tmp[6];
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
               << flagcxReduceTriggerOffState;
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

flagcxResult_t flagcxFifo::flagcxRedFifoInit(size_t numTriggers) {
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  const size_t headerBytes = flagcxRedFifoIdxData * sizeof(uint64_t);
  const size_t triggerBytes = numTriggers * sizeof(flagcxReduceTrigger);
  uint64_t header[flagcxRedFifoIdxData] = {static_cast<uint64_t>(numTriggers),
                                           0};

  if (buffer != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemDevice,
                                          NULL));
    buffer = NULL;
  }
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      (void **)&buffer, headerBytes + triggerBytes, flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemset(buffer, 0, headerBytes + triggerBytes,
                                          flagcxMemDevice, NULL));
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(buffer, header, headerBytes,
                                          flagcxMemcpyHostToDevice, NULL,
                                          NULL));
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  if (buffer != NULL) {
    FLAGCXCHECK(deviceAdaptor->deviceFree((void *)buffer, flagcxMemDevice,
                                          NULL));
    buffer = NULL;
  }
  return flagcxSuccess;
}
