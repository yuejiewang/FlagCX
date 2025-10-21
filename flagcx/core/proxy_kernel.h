#include "check.h"
#include "device.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "comm.h"
#include <unistd.h>

struct flagcxProxyKernelState {
  pthread_t thread;
  flagcxFifo_t fifo;
};

struct flagcxProxyKernelServiceArgs {
  flagcxComm_t comm;
  flagcxStream_t stream;
};
typedef struct flagcxProxyKernelServiceArgs *flagcxProxyKernelServiceArgs_t;
void *flagcxProxyKernelService(void *args);
flagcxResult_t flagcxProxyKernelInit(struct flagcxHeteroComm *comm);
flagcxResult_t flagcxProxyKernelDestroy(struct flagcxHeteroComm *comm);