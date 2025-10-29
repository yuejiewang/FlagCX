#ifndef FLAGCX_KERNEL_H_
#define FLAGCX_KERNEL_H_

#include "cuda.h"
#include "cuda_runtime.h"
#include "flagcx.h"

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
constexpr unsigned int flagcxDeviceTriggerBitsReady = 1;
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
  } value;
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
           flagcxDeviceTriggerBitsReady -
           flagcxDeviceTriggerBitsFifoReserved); // ensure 64-bit alignment
    uint64_t reserved : flagcxDeviceTriggerBitsFifoReserved;
    uint64_t ready : flagcxDeviceTriggerBitsReady;
  } fields;
} flagcxDeviceTrigger;
typedef flagcxDeviceTrigger *flagcxDeviceTrigger_t;

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
typedef flagcxReduceTrigger *flagcxReduceTrigger_t;

struct flagcxFifo {
  int32_t capacity;
  int32_t *produced;
  int32_t *consumed;
  int32_t *terminate;
  uint64_t *buffer;

public:
  __host__ flagcxResult_t initFifo(int32_t capacity_);
  __host__ flagcxResult_t freeFifo();
  // device-producer + host-consumer APIs
  __device__ flagcxResult_t enqueue(flagcxDeviceTrigger trigger);
  __host__ flagcxResult_t dequeue(flagcxDeviceTrigger_t trigger);
  // host-producer + device-consumer APIs
  __host__ flagcxResult_t enqueue(flagcxReduceTrigger trigger);
  __device__ flagcxResult_t dequeue(flagcxReduceTrigger_t trigger);
};
typedef struct flagcxFifo *flagcxFifo_t;

__device__ size_t getFlagcxDataTypeSizeDevice(flagcxDataType_t dtype);

__device__ flagcxResult_t flagcxDeviceSend(const void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxFifo_t fifo);
__device__ flagcxResult_t flagcxDeviceRecv(void *sendbuff, size_t count,
                                           flagcxDataType_t datatype, int peer,
                                           flagcxFifo_t fifo);
__device__ flagcxResult_t flagcxDeviceWait(flagcxFifo_t fifo);
__device__ flagcxResult_t flagcxDeviceTerm(flagcxFifo_t fifo);
__global__ void flagcxCollectiveKernel(flagcxFifo_t q); // TBD
void flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                   flagcxDataType_t datatype, int sendPeer, int recvPeer,
                   flagcxComm_t comm, flagcxStream_t stream);
#endif
