#include "flagcx.h"
#include "flagcx_kernel.h"
#include "global_comm.h"
#include "comm.h"
FLAGCX_GLOBAL_DECORATOR void flagcxP2pKernel(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype,
                              int sendPeer, int recvPeer, void *fifoBuffer) {
  int tid = threadIdx.x;
  printf("FlagCX P2P Demo Kernel launched with tid %d\n", tid);
  if (tid == 0) {
    for (int i = 0; i < 1; i++) {
      const void *sendaddr = static_cast<const void *>(
          static_cast<char *>(const_cast<void *>(sendbuff)));
          // static_cast<char *>(const_cast<void *>(sendbuff)) +
          // count / 2 * i * getFlagcxDataTypeSizeDevice(datatype));
      flagcxDeviceSend(sendaddr, count, datatype, sendPeer,
                       fifoBuffer);
    }
    printf("FlagCX P2P Demo Kernel flagcxDeviceSend with peer %d\n", sendPeer);
    for (int i = 0; i < 1; i++) {
      void *recvaddr = static_cast<void *>(
          static_cast<char *>(recvbuff)); 
          // static_cast<char *>(recvbuff) +
          // count / 2 * i * getFlagcxDataTypeSizeDevice(datatype));
      flagcxDeviceRecv(recvaddr, count, datatype, recvPeer,
                       fifoBuffer);
    }
    printf("FlagCX P2P Demo Kernel flagcxDeviceRecv with peer %d\n", recvPeer);
    flagcxDeviceTerm(fifoBuffer);
    printf("FlagCX P2P Demo Kernel flagcxDeviceTerm\n");
    flagcxDeviceWait(fifoBuffer);
    printf("FlagCX P2P Demo Kernel flagcxDeviceWait\n");
  }
}

void flagcxP2pDemo(const void *sendbuff, void *recvbuff, size_t count,
                              flagcxDataType_t datatype, int sendPeer,
                              int recvPeer, flagcxComm_t comm,
                              flagcxStream_t stream) {
  flagcxP2pKernel<<<1, 1, 0, *(FLAGCX_DEVICE_STREAM_PTR)stream>>>(
      sendbuff, recvbuff, count, datatype, sendPeer, recvPeer, comm->hetero_comm->fifoBuffer);
}
