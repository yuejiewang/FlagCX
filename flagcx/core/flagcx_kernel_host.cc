#include "comm.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

FLAGCX_HOST_DECORATOR flagcxResult_t dequeue(void *fifoBuffer,
                                             flagcxDeviceTrigger_t trigger) {
  int idx = -1;
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int capacity = buffer[0];
  int distance = buffer[2] - buffer[1];
  if (distance > 0) {
    idx = buffer[1] % capacity;
    buffer[1] = buffer[1] + 1;
    __sync_synchronize();
  }
  if (idx > -1) {
    memcpy((void *)trigger,
           (void *)(buffer + 3 +
                    sizeof(flagcxDeviceTrigger) / sizeof(uint64_t) * idx),
           sizeof(flagcxDeviceTrigger));
    printf("host dequeue from buff 0x%016lx, cons=%lu, prod=%lu, idx %d, addr "
           "0x%016lx, fst 0x%016lx, snd 0x%016lx\n",
           (uintptr_t)buffer, buffer[1], buffer[2], idx,
           (uintptr_t)(buffer + 3 +
                       sizeof(flagcxDeviceTrigger) / sizeof(uint64_t) * idx),
           trigger->fst, trigger->snd);
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}

// __host__ flagcxResult_t flagcxFifo::enqueue(flagcxReduceTrigger trigger) {
//   // to be implemented
//   return flagcxNotSupported;
// }
