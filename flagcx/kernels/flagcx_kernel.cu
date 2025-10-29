#include "adaptor.h"
#include "comm.h"
#include "flagcx.h"
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

__device__ size_t getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      return 0;
  }
}

__device__ flagcxResult_t flagcxDeviceSend(const void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxFifo_t fifo) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = (uint64_t)(sendbuff);
  trigger.fields.count = count;
  trigger.fields.peerRank = peer;
  trigger.fields.datatype = datatype;
  trigger.fields.type = flagcxDevicePrimSend;
  trigger.fields.ready = 0;
  fifo->enqueue(trigger);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceRecv(void *recvbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxFifo_t fifo) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = (uint64_t)(recvbuff);
  trigger.fields.count = count;
  trigger.fields.peerRank = peer;
  trigger.fields.datatype = datatype;
  trigger.fields.type = flagcxDevicePrimRecv;
  trigger.fields.ready = 0;
  fifo->enqueue(trigger);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceWait(flagcxFifo_t fifo) {
  int curr_p = __ldg(fifo->produced);
  int curr_c = __ldg(fifo->consumed);
  int iter = 0;
  while (curr_p > curr_c) {
    // curr_p = __ldg(comm->proxyKernelState->fifo->produced);
    // check a fixed point, not updating `produced` index
    curr_c = __ldg(fifo->consumed);
    spin_backoff(iter);
    iter++;
  }
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceTerm(flagcxFifo_t fifo) {
  flagcxDeviceTrigger trigger;
  trigger.fields.addr = 0;
  trigger.fields.count = 0;
  trigger.fields.peerRank = 0;
  trigger.fields.datatype = 0;
  trigger.fields.type = flagcxDevicePrimTerm;
  trigger.fields.ready = 0;
  fifo->enqueue(trigger);
  return flagcxSuccess;
}

__host__ flagcxResult_t flagcxFifo::initFifo(int32_t capacity_) {
  TRACE(FLAGCX_P2P, "Initialize FIFO...");
  void *tmp_buff;
  deviceAdaptor->deviceMalloc(&tmp_buff, capacity * sizeof(flagcxDeviceTrigger),
                              flagcxMemHost, NULL);
  buffer = (uint64_t *)tmp_buff;
  void *tmp_p;
  deviceAdaptor->deviceMalloc(&tmp_p, sizeof(int32_t), flagcxMemHost, NULL);
  produced = (int32_t *)tmp_p;
  void *tmp_c;
  deviceAdaptor->deviceMalloc(&tmp_c, sizeof(int32_t), flagcxMemHost, NULL);
  consumed = (int32_t *)tmp_c;
  void *tmp_term;
  deviceAdaptor->deviceMalloc(&tmp_term, sizeof(int32_t), flagcxMemHost, NULL);
  terminate = (int32_t *)tmp_term;
  produced[0] = -1;
  consumed[0] = -1;
  terminate[0] = -1;
  return flagcxSuccess;
}

__host__ flagcxResult_t flagcxFifo::freeFifo() {
  deviceAdaptor->deviceFree(buffer, flagcxMemHost, NULL);
  deviceAdaptor->deviceFree(produced, flagcxMemHost, NULL);
  deviceAdaptor->deviceFree(consumed, flagcxMemHost, NULL);
  deviceAdaptor->deviceFree(terminate, flagcxMemHost, NULL);
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
        idx = (old_p + 1) % capacity;
        break;
      }
    }
  }
  *(buffer + 2 * idx) = trigger.value.fst;
  *(buffer + 2 * idx + 1) = trigger.value.snd;
  *(buffer + 2 * idx + 1) |= 1ULL;
  return flagcxSuccess;
}

__host__ flagcxResult_t flagcxFifo::dequeue(flagcxDeviceTrigger_t trigger) {
  /*
  int idx = -1;
  while (true) {
    int old_c = consumed[0];
    int old_p = produced[0];
    if (old_c < old_p) {
      int prev = atomicCAS(consumed, old_c, old_c + 1);
      if (prev == old_c) {
        idx = old_c + 1;
        break;
      }
    }
  }
  */
  int idx = -1;
  int old_c = consumed[0];
  int old_p = produced[0];
  while (true) {
    if (old_p < produced[0]) {
      TRACE(FLAGCX_P2P, "fifo dequeue loop, produced = %d", old_p);
    }
    if (old_p > old_c) {
      idx = (old_c + 1) % capacity;
      while (true) {
        // check if producer write is complete
        TRACE(FLAGCX_P2P, "write check loop");
        if (*(buffer + 2 * idx + 1) & 1ULL) {
          break;
        }
      }
      break;
    }
    old_c = consumed[0];
    old_p = produced[0];
   }
  (*trigger).value.fst = *(buffer + 2 * idx);
  (*trigger).value.snd = *(buffer + 2 * idx + 1);
  *(buffer + 2 * idx) = 0ULL;
  *(buffer + 2 * idx + 1) = 0ULL;
  consumed[0]++;
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
