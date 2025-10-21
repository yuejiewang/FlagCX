#include "flagcx.h"
#include <cuda.h>
#include <cuda_runtime.h>

typedef enum {
  flagcxDevicePrimSend = 0,
  flagcxDevicePrimRecv = 1,
  flagcxDevicePrimTerm = 2
} flagcxDevicePrim;

constexpr unsigned int flagcxDeviceTriggerBitsAddr = 64;
// constexpr unsigned int flagcxDeviceTriggerBitsOffset = 32;
constexpr unsigned int flagcxDeviceTriggerBitsCount = 32;
constexpr unsigned int flagcxDeviceTriggerBitsPeerRank = 20;
constexpr unsigned int flagcxDeviceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxDeviceTriggerBitsPrim = 4;
constexpr unsigned int flagcxDeviceTriggerBitsFifoReserved = 1;

constexpr unsigned int flagcxReduceTriggerBitsAddr = 64;
constexpr unsigned int flagcxReduceTriggerBitsCount = 32;
constexpr unsigned int flagcxReduceTriggerBitsNThreads = 16;
constexpr unsigned int flagcxReduceTriggerBitsDatatype = 4;
constexpr unsigned int flagcxReduceTriggerBitsRedop = 4;
constexpr unsigned int flagcxReduceTriggerBitsFifoReserved = 1;

typedef union alignas(16) {
  struct {
    uint64_t fst;
    uint64_t snd;
  };
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t addr : flagcxDeviceTriggerBitsAddr;
    // uint64_t offset : flagcxDeviceTriggerBitsOffset;
    // Second 64 bits: value[1]
    uint64_t count : flagcxDeviceTriggerBitsCount;
    uint64_t peerRank : flagcxDeviceTriggerBitsPeerRank;
    uint64_t datatype : flagcxDeviceTriggerBitsDatatype;
    uint64_t type : flagcxDeviceTriggerBitsPrim;
    uint64_t
        : (64 - flagcxDeviceTriggerBitsCount - flagcxDeviceTriggerBitsPeerRank -
           flagcxDeviceTriggerBitsDatatype - flagcxDeviceTriggerBitsPrim -
           flagcxDeviceTriggerBitsFifoReserved); // ensure 64-bit alignment
    uint64_t reserved : flagcxDeviceTriggerBitsFifoReserved;
  } fields;
} flagcxDeviceTrigger;
typedef union flagcxDeviceTrigger *flagcxDeviceTrigger_t;

typedef union alignas(16) {
  uint64_t value[4];
  struct {
    // First 64 bits: value[0]
    uint64_t addr1 : flagcxReduceTriggerBitsAddr;
    // Second 64 bits: value[1]
    uint64_t addr2 : flagcxReduceTriggerBitsAddr;
    // Third 64 bits: value[2]
    uint64_t addr3 : flagcxReduceTriggerBitsAddr;
    // Last 64 bits: value[3]
    uint64_t count : flagcxReduceTriggerBitsCount;
    uint64_t nthreads : flagcxReduceTriggerBitsNThreads;
    uint64_t datatype : flagcxReduceTriggerBitsDatatype;
    uint64_t redop : flagcxReduceTriggerBitsRedop;
    uint64_t
        : (64 - flagcxReduceTriggerBitsCount - flagcxReduceTriggerBitsNThreads -
           flagcxReduceTriggerBitsDatatype - flagcxReduceTriggerBitsRedop -
           flagcxReduceTriggerBitsFifoReserved);
    uint64_t reserved : flagcxReduceTriggerBitsFifoReserved;
  } fields;
} flagcxReduceTrigger;
typedef union flagcxReduceTrigger *flagcxReduceTrigger_t;

struct flagcxFifo {
  int32_t capacity;
  int32_t *produced;
  int32_t *consumed;
  int32_t *terminate;
  uint64_t *buffer;

public:
  flagcxFifo(int32_t capacity_, int32_t *produced_, int32_t *consumed_,
             int32_t *terminate_, uint64_t *buffer_)
      : capacity(capacity_), produced(produced_), consumed(consumed_),
        terminate(terminate_), buffer(buffer_) {
    produced[0] = -1;
    consumed[0] = -1;
    terminate[0] = -1;
  }
  ~flagcxFifo() {
    free(produced);
    free(consumed);
    free(terminate);
    free(buffer);
  }
  // device-producer + host-consumer APIs
  __device__ flagcxResult_t enqueue(flagcxDeviceTrigger trigger);
  __host__ flagcxResult_t dequeue(flagcxDeviceTrigger_t trigger);
  // host-producer + device-consumer APIs
  __host__ flagcxResult_t enqueue(flagcxReduceTrigger trigger);
  __device__ flagcxResult_t dequeue(flagcxReduceTrigger_t trigger);
};
typedef struct flagcxFifo *flagcxFifo_t;

__device__ flagcxResult_t flagcxDeviceSend(const void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxComm_t comm);
__device__ flagcxResult_t flagcxDeviceRecv(void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxComm_t comm);
__device__ flagcxResult_t flagcxDeviceWait(flagcxComm_t comm);
__device__ flagcxResult_t flagcxDeviceTerm(flagcxComm_t comm);

__global__ void flagcxCollectiveKernel(flagcxFifo_t q); // TBD

__device__ __forceinline__ void spin_backoff(int iter) {
  int delay = 1 << min(15, iter);
#if __CUDA_ARCH__ >= 700
  __nanosleep(delay);
#else
  uint64_t start = clock64();
  while (clock64() - start < (uint64_t)delay) { /* spin */
  }
#endif
}
