#ifdef USE_ASCEND_ADAPTOR
#include "acl/acl.h"
#include "flagcx.h"
#include "hccl/hccl.h"
#include <map>
struct flagcxInnerDevComm {};

struct flagcxInnerComm {
  HcclComm base;
};

struct flagcxStream {
  aclrtStream base;
};

struct flagcxEvent {
  aclrtEvent base;
};

#define FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES 64
#define FLAGCX_ASCEND_IPC_KEY_BUFFER_BYTES 65

// ACL IPC keys are 64-byte strings plus a trailing NUL. FlagCX transports
// exactly 64 bytes of opaque IPC handle data, so the terminator is rebuilt
// locally before calling an ACL import/close API.
struct flagcxIpcMemHandle {
  char key[FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES];
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != ACL_SUCCESS)                                                    \
      return flagcxUnhandledDeviceError;                                       \
  }
#endif // USE_ASCEND_ADAPTOR
