#include "flagcx_kernel.h"

__device__ __forceinline__ void spin_backoff(int iter) {
  int delay = 1 << (iter < 15 ? iter : 15);
#if __CUDA_ARCH__ >= 700
  __nanosleep(delay);
#else
  uint64_t start = clock64();
  while (clock64() - start < (uint64_t)delay) { /* spin */
  }
#endif
}

__device__ flagcxResult_t flagcxDeviceSend(const void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxHeteroComm_t comm) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = const_cast<void *>sendbuff;
  trigger.fields.count = count;
  trigger.fields.peerRank = peer;
  trigger.fields.datatype = datatype;
  trigger.fields.type = flagcxDevicePrimSend;
  comm->proxyKernelState->fifo->enqueue(trigger);
  return flagcxSuccess;
}
__device__ flagcxResult_t flagcxDeviceRecv(void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxHeteroComm_t comm) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = const_cast<void *> recvbuff;
  trigger.fields.count = count;
  trigger.fields.peerRank = peer;
  trigger.fields.datatype = datatype;
  trigger.fields.type = flagcxDevicePrimRecv;
  comm->proxyKernelState->fifo->enqueue(trigger);
  return flagcxSuccess;
}
__device__ flagcxResult_t flagcxDeviceWait(flagcxHeteroComm_t comm) {
  int curr_p = __ldg(comm->fifo->produced[0]);
  int curr_c = __ldg(comm->fifo->consumed[0]);
  int iter = 0;
  while (curr_p > curr_c) {
    // curr_p = __ldg(comm->fifo->produced[0]);
    // check a fixed point, not updating `produced` index
    curr_c = __ldg(comm->fifo->consumed[0]);
    spin_backoff(iter);
    iter++;
  }
  return flagcxSuccess;
}
__device__ flagcxResult_t flagcxDeviceTerm(flagcxHeteroComm_t comm) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = 0;
  trigger.fields.count = 0;
  trigger.fields.peerRank = 0;
  trigger.fields.datatype = 0;
  trigger.fields.type = flagcxDevicePrimTerm;
  comm->proxyKernelState->fifo->enqueue(trigger);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxFifo::enqueue(flagcxDeviceTrigger trigger) {
  int idx = -1;
  while (true) {
    int old_c = consumed[0];
    int old_p = produced[0];
    if (old_p - old_c < capacity) {
      int prev = atomicCAS(produced, old_p, old_p + 1);
      if (prev == old_p) {
        idx = old_p;
        break;
      }
    }
  }
  buffer[idx] = trigger;
  return flagcxSuccess;
}

__host__ flagcxResult_t flagcxFifo::dequeue(flagcxDeviceTrigger_t trigger) {
  *trigger = buffer[++consumed[0]];
  return flagcxSuccess;
}

__host__ flagcxResult_t flagcxFifo::enqueue(flagcxReduceTrigger trigger) {
  // to be implemented
  return flagcxNotSupported;
}

__device__ flagcxResult_t flagcxFifo::dequeue(flagcxReduceTrigger_t trigger) {
  // to be implemented
  return flagcxNotSupported;
}
