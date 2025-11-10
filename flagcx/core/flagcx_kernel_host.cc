#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  uint64_t capacity = buffer[0];
  uint64_t cons = buffer[1];
  uint64_t prod = buffer[2];
  int distance = prod - cons;
  if (distance > 0) {
    idx = cons % capacity;
    memcpy((void *)trigger,
           (void *)(buffer + 3 +
                    sizeof(flagcxDeviceTrigger) / sizeof(uint64_t) * idx),
           sizeof(flagcxDeviceTrigger));
    __sync_synchronize();
    buffer[1] = buffer[1] + 1;
    // __sync_synchronize();
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}

// __host__ flagcxResult_t flagcxFifo::enqueue(flagcxReduceTrigger trigger) {
//   // to be implemented
//   return flagcxNotSupported;
// }
