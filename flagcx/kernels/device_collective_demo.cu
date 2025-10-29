#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
#include "proxy.h"
#include "nvidia_adaptor.h"

__global__ void flagcxP2pKernel(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int sendPeer, int recvPeer, flagcxFifo_t fifo) {
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 16; i++) {
      const void *sendaddr = static_cast<const void *>(
          static_cast<char *>(const_cast<void *>(sendbuff)) +
          count / 16 * i * getFlagcxDataTypeSizeDevice(datatype));
      flagcxDeviceSend(sendaddr, count / 16, datatype, sendPeer, fifo);
    }
    for (int i = 0; i < 16; i++) {
      void *recvaddr = static_cast<void *>(
          static_cast<char *>(recvbuff) +
          count / 16 * i * getFlagcxDataTypeSizeDevice(datatype));
      flagcxDeviceRecv(recvaddr, count / 16, datatype, recvPeer, fifo);
    }
    flagcxDeviceTerm(fifo);
    flagcxDeviceWait(fifo);
  }
}

void flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                   flagcxDataType_t datatype, int sendPeer, int recvPeer,
                   flagcxComm_t comm, flagcxStream_t stream) {
  TRACE(FLAGCX_P2P, "rank %d launch P2P kernel", comm->rank);
  flagcxP2pKernel<<<1, 1, 0, stream->base>>>(
      sendbuff, recvbuff, count, datatype, sendPeer, recvPeer,
      comm->hetero_comm->proxyState->proxyKernelState->fifo);
  deviceAdaptor->streamSynchronize(stream);
  TRACE(FLAGCX_P2P, "rank %d P2P kernel terminate", comm->rank);
}
