/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 ************************************************************************/

#ifdef USE_KUNLUNXIN_ADAPTOR

#include <map>

#include <bkcl.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "flagcx.h"

struct flagcxInnerComm {
    BKCLContext_t base;
};

struct flagcxStream {
    cudaStream_t base;
};

struct flagcxEvent {
    cudaEvent_t base;
};

#define DEVCHECK(func)                                                         \
  {                                                                            \
    int ret = func;                                                            \
    if (ret != cudaSuccess)                                                      \
      return flagcxUnhandledDeviceError;                                       \
  }

#endif // USE_KUNLUNXIN_ADAPTOR