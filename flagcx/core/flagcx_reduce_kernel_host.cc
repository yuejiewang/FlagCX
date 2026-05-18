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

flagcxResult_t flagcxFifo::flagcxRedFifoInit(size_t capacity) {
  TRACE(FLAGCX_INIT, "flagcxRedFifoInit called");
  size_t bytes = flagcxRedFifoIdxData * sizeof(uint64_t) +
                 capacity * sizeof(flagcxReduceTrigger);
  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&buffer, bytes,
                                          flagcxMemDevice, NULL));
  FLAGCXCHECK(
      deviceAdaptor->deviceMemset(buffer, 0, bytes, flagcxMemDevice, NULL));
  return flagcxSuccess;
}

flagcxResult_t flagcxFifo::flagcxRedFifoDestroy() {
  INFO(FLAGCX_KERNEL, "flagcxRedFifoDestroy called");
  if (buffer != NULL) {
    FLAGCXCHECK(
        deviceAdaptor->deviceFree((void *)buffer, flagcxMemDevice, NULL));
    buffer = NULL;
  }
  return flagcxSuccess;
}
