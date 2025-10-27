#include "proxy_kernel.h"

void *flagcxProxyKernelService(void *args) {
  flagcxProxyKernelServiceArgs_t argptr = (flagcxProxyKernelServiceArgs_t)args;
  struct flagcxHeteroComm *comm = argptr->comm;
  flagcxFifo_t fifo = comm->proxyKernelState->fifo;
  flagcxStream_t stream = argptr->stream;
  int groupCount = 0;
  flagcxDeviceTrigger_t ptr;
  flagcxCalloc(&ptr, sizeof(flagcxDeviceTrigger));
  while (true) {
    if (groupCount == 0) {
      flagcxHeteroGroupStart();
      TRACE(FLAGCX_P2P,
            "rank=%d flagcxHeteroGroupStart called by proxyKernelService.",
            comm->rank);
      groupCount++;
    }
    fifo->dequeue(ptr);
    flagcxDeviceTrigger trigger = *ptr;
    switch (trigger.fields.type) {
      case flagcxDevicePrimSend:
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimSend called by proxyKernelService.",
              comm->rank);
        flagcxHeteroSend((const void *)(uintptr_t)(trigger.fields.addr),
                         trigger.fields.count,
                         (flagcxDataType_t)(trigger.fields.datatype),
                         trigger.fields.peerRank, comm, stream);
        break;
      case flagcxDevicePrimRecv:
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimRecv called by proxyKernelService.",
              comm->rank);
        flagcxHeteroRecv((void *)(uintptr_t)(trigger.fields.addr),
                         trigger.fields.count,
                         (flagcxDataType_t)(trigger.fields.datatype),
                         trigger.fields.peerRank, comm, stream);
        break;
      case flagcxDevicePrimTerm:
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxHeteroGroupEnd called by proxyKernelService.",
              comm->rank);
        flagcxHeteroGroupEnd();
        groupCount--;
        break;
      default:
        break;
    }
  }
  free(ptr);
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
