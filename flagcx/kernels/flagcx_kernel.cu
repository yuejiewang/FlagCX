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
                                           void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)sendbuff), count, peer, datatype, flagcxDevicePrimSend);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceRecv(void *recvbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           void *fifoBuffer) {
  enqueue(fifoBuffer, (uint64_t)((uintptr_t)recvbuff), count, peer, datatype, flagcxDevicePrimRecv);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceTerm(void *fifoBuffer) {
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimTerm);
  return flagcxSuccess;
}

__device__ flagcxResult_t flagcxDeviceWait(void *fifoBuffer) {
  enqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimWait);
  unsigned long long int *buffer = (unsigned long long int *)fifoBuffer;
  int curr_c = buffer[1];
  int curr_p = buffer[2];
  int iter = 0;
  while (curr_p > curr_c) {
    spin_backoff(iter);
    iter++;
    __threadfence_system();
    curr_c = buffer[1];
    // printf("flagcxDeviceWait spinning... curr_c=%d, curr_p=%d, iter=%d\n", curr_c, curr_p, iter);
  }
  printf("Exit wait\n");
  return flagcxSuccess;
}

__device__ flagcxResult_t enqueue(void *fifoBuffer, uint64_t addr, uint64_t count, uint64_t peerRank, uint64_t datatype, uint64_t type) {
  int idx = -1;
  unsigned long long int *buffer = (unsigned long long int *)fifoBuffer;
  int capacity = buffer[0];
  int old_c = buffer[1];
  int old_p = buffer[2];
  // printf("Enter Enqueue capacity=%d, consumed=%d, produced=%d\n", capacity, (int)buffer[1], (int)buffer[2]);
  while (true) {
    old_c = buffer[1];
    old_p = buffer[2];
    int n = old_p - old_c;
    if (n < 0) n += capacity;
    if (n < capacity) {
      int prev = atomicCAS(buffer + 2, old_p, (old_p + 1) % capacity);
      if (prev == old_p) {
        idx = old_p;
        break;
      }
    }
  }
  *(buffer + 3 + 5 * idx) = addr;
  *(buffer + 3 + 5 * idx + 1) = count;
  *(buffer + 3 + 5 * idx + 2) = peerRank;
  *(buffer + 3 + 5 * idx + 3) = datatype;
  *(buffer + 3 + 5 * idx + 4) = type;
  __threadfence_system();
  // printf("Enqueue capacity=%d, consumed=%d, produced=%d\n", capacity, (int)buffer[1], (int)buffer[2]);
  return flagcxSuccess;
}

__host__ flagcxResult_t dequeue(void *fifoBuffer, flagcxDeviceTrigger_t trigger) {
  int idx = -1;
  unsigned long long int *buffer = (unsigned long long int *)fifoBuffer;
  int capacity = buffer[0];
  int old_c = buffer[1];
  int old_p = buffer[2];
  if (old_c != old_p) {
    // buffer[1] = (old_c + 1) % capacity;
    idx = old_c;
  }
  if (idx > -1) {
    memcpy((void *)trigger, (void *)(buffer + 3 + 5 * idx), sizeof(flagcxDeviceTrigger));
  } else {
    memset((void *)trigger, 0, sizeof(flagcxDeviceTrigger));
  }
  return flagcxSuccess;
}

// __host__ flagcxResult_t flagcxFifo::enqueue(flagcxReduceTrigger trigger) {
//   // to be implemented
//   return flagcxNotSupported;
// }

// __device__ flagcxResult_t flagcxFifo::dequeue(flagcxReduceTrigger_t trigger) {
//   // to be implemented
//   return flagcxNotSupported;
// }
