#include "flagcx_kernel.h"

__global__ void flagcxP2pDemo(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype, int peer,
                              flagcxComm_t comm) {
  int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < 16; i++) {
      const void *sendaddr = const_cast<const void *>(
          const_cast<char *>(sendbuff) +
          count / 16 * i * getFlagcxDataTypeSize(datatype));
      flagcxDeviceSend(sendaddr, count / 16, datatype, peer, comm);
    }
    for (int i = 0; i < 16; i++) {
      void *recvaddr =
          const_cast<void *>(const_cast<char *>(recvbuff) +
                             count / 16 * i * getFlagcxDataTypeSize(datatype));
      flagcxDeviceRecv(recvaddr, count / 16, datatype, peer, comm);
    }
    flagcxDeviceTerm(comm);
    flagcxDeviceWait(comm);
  }
}
