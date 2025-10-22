#include "proxy_kernel.h"

void *flagcxProxyKernelService(void *args) {
  flagcxProxyKernelServiceArgs_t ptr = (flagcxProxyKernelServiceArgs_t)args;
  struct flagcxHeteroComm *comm = ptr->comm;
  flagcxFifo_t fifo = comm->proxyKernelState->fifo;
  flagcxStream_t stream = ptr->stream;
  int groupCount = 0;
  while (true) {
    if (groupCount == 0) {
      flagcxHeteroGroupStart();
      groupCount++;
    }
    flagcxDeviceTrigger_t ptr;
    flagcxCalloc(&ptr, sizeof(flagcxDeviceTrigger));
    fifo->dequeue(ptr);
    flagcxDeviceTrigger trigger = *ptr;
    switch (trigger.fields.type) {
      case flagcxDevicePrimSend:
        flagcxSend((const void *)(uintptr_t)(trigger.fields.addr),
                   trigger.fields.count,
                   (flagcxDataType_t)(trigger.fields.datatype),
                   trigger.fields.peerRank, comm, stream);
        break;
      case flagcxDevicePrimRecv:
        flagcxRecv((void *)(uintptr_t)(trigger.fields.addr),
                   trigger.fields.count,
                   (flagcxDataType_t)(trigger.fields.datatype),
                   trigger.fields.peerRank, comm, stream);
        break;
      case flagcxDevicePrimTerm:
        flagcxHeteroGroupEnd();
        groupCount--;
        break;
      default:
        break;
    }
  }
}

flagcxResult_t flagcxProxyKernelInit(struct flagcxHeteroComm *comm) {
  INFO(FLAGCX_INIT, "rank=%d flagcxProxyKernelInit called.", comm->rank);
  pthread_create(&comm->proxyKernelState->thread, NULL,
                 flagcxProxyKernelService, (void *)comm);
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyKernelDestroy(struct flagcxHeteroComm *comm) {
  pthread_join(comm->proxyKernelState->thread, nullptr);
  comm->proxyKernelState->fifo->~flagcxFifo();
  free(comm->proxyKernelState->fifo);
  return flagcxSuccess;
}
