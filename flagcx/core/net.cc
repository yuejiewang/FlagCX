#include "net.h"
#include "adaptor.h"
#include "device.h"
#include "proxy.h"

flagcxResult_t flagcxProxySend(sendNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (deviceKernel) {
    deviceAdaptor->deviceMemcpy(args->hEventReady,args->dEventReady, 1,
                                flagcxMemcpyDeviceToHost, resources->cpStream,
                                NULL);
  }
  if (!__atomic_load_n(&args->hEventReady, __ATOMIC_RELAXED))
    return flagcxSuccess;
  if (args->transmitted < args->chunkSteps) {
    int stepMask = args->sendStepMask;

    if (args->waitCopy < args->chunkSteps &&
        args->waitCopy - args->transmitted < MAXSTEPS) {
      int step = args->waitCopy & stepMask;
      args->subs[step].stepSize =
          std::min(args->chunkSize, size - args->totalCopySize);
      args->subs[step].stepBuff = resources->buffers[0] + (CHUNKSIZE * step);
      deviceAdaptor->deviceMemcpy(
          args->subs[step].stepBuff, (char *)data + args->totalCopySize,
          args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
          resources->cpStream, args->subs[step].copyArgs);
      deviceAdaptor->eventRecord(resources->cpEvents[step],
                                 resources->cpStream);
      args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->copied < args->waitCopy) {
      int step = args->copied & stepMask;
      if (deviceAdaptor->eventQuery(resources->cpEvents[step]) ==
          flagcxSuccess) {
        args->copied++;
      }
    }

    if (args->posted < args->copied) {
      void *req = NULL;
      flagcxNetIb.isend(resources->netSendComm,
                        args->subs[args->posted & stepMask].stepBuff,
                        args->subs[args->posted & stepMask].stepSize, 0,
                        resources->mhandles[0], &req);
      if (req) {
        args->subs[args->posted++ & stepMask].requests[0] = req;
      }
    }

    if (args->transmitted < args->posted) {
      void *req = args->subs[args->transmitted & stepMask].requests[0];
      int done = 0, sizes;
      flagcxNetIb.test(req, &done, &sizes);
      if (done) {
        args->transmitted++;
      }
    }
  } else {
    __atomic_store_n(args->hlArgs, 1, __ATOMIC_RELAXED);
    args->done = true;
    if (deviceKernel) {
      deviceAdaptor->deviceMemcpy(args->dlArgs, args->hlArgs, 1,
                                  flagcxMemcpyHostToDevice, resources->cpStream,
                                  NULL);
    }
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxProxyRecv(recvNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (deviceKernel) {
    deviceAdaptor->deviceMemcpy(args->hEventReady,args->dEventReady, 1,
                                flagcxMemcpyDeviceToHost, resources->cpStream,
                                NULL);
  }
  if (!__atomic_load_n(&args->hEventReady, __ATOMIC_RELAXED))
    return flagcxSuccess;
  if (args->copied < args->chunkSteps) {
    int stepMask = args->sendStepMask;
    if (args->posted < args->chunkSteps &&
        args->posted - args->copied < MAXSTEPS) {
      int tags[8] = {0};
      void *req = NULL;
      args->subs[args->posted & stepMask].stepSize =
          std::min(args->chunkSize, size - args->totalPostSize);
      args->subs[args->posted & stepMask].stepBuff =
          resources->buffers[0] + CHUNKSIZE * (args->posted & stepMask);
      flagcxNetIb.irecv(resources->netRecvComm, 1,
                        &args->subs[args->posted & stepMask].stepBuff,
                        (int *)&args->subs[args->posted & stepMask].stepSize,
                        tags, resources->mhandles, &req);
      if (req) {
        args->subs[args->posted & stepMask].requests[0] = req;
        args->totalPostSize += args->subs[args->posted++ & stepMask].stepSize;
      }
    }

    if (args->transmitted < args->posted) {
      void *req = args->subs[args->transmitted & stepMask].requests[0];
      int done = 0, sizes;
      flagcxNetIb.test(req, &done, &sizes);
      if (done) {
        args->transmitted++;
      }
    }

    if (args->postFlush < args->transmitted) {
      void *req = NULL;
      void *allData[] = {args->subs[args->postFlush & stepMask].stepBuff};
      flagcxNetIb.iflush(resources->netRecvComm, 1, allData,
                         &args->subs[args->postFlush & stepMask].stepSize,
                         resources->mhandles, &req);
      if (req) {
        args->subs[args->postFlush++ & stepMask].requests[0] = req;
      }
    }

    if (args->flushed < args->postFlush) {
      void *req = args->subs[args->flushed & stepMask].requests[0];
      int done = 0, sizes;
      flagcxNetIb.test(req, &done, &sizes);
      if (done) {
        args->flushed++;
      }
    }

    if (args->waitCopy < args->flushed) {
      int step = args->waitCopy & stepMask;
      deviceAdaptor->deviceMemcpy(
          (char *)data + args->totalCopySize, args->subs[step].stepBuff,
          args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
          resources->cpStream, args->subs[step].copyArgs);
      deviceAdaptor->eventRecord(resources->cpEvents[step],
                                 resources->cpStream);
      args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->copied < args->waitCopy) {
      int step = args->copied & stepMask;
      if (deviceAdaptor->eventQuery(resources->cpEvents[step]) ==
          flagcxSuccess) {
        args->copied++;
      }
    }

  } else {
    __atomic_store_n(args->hlArgs, 1, __ATOMIC_RELAXED);
    args->done = true;
    if (deviceKernel) {
      deviceAdaptor->deviceMemcpy(args->dlArgs, args->hlArgs, 1,
                                  flagcxMemcpyHostToDevice, resources->cpStream,
                                  NULL);
    }
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxSendProxyFree(sendNetResources *resources) {
  flagcxNetIb.deregMr(resources->netSendComm, resources->mhandles[0]);
  flagcxNetIb.closeSend(resources->netSendComm);
  deviceAdaptor->gdrMemFree(resources->buffers[0], NULL);
  for (int s = 0; s < MAXSTEPS; s++) {
    deviceAdaptor->eventDestroy(resources->cpEvents[s]);
  }
  deviceAdaptor->streamDestroy(resources->cpStream);
  return flagcxSuccess;
}

flagcxResult_t flagcxRecvProxyFree(recvNetResources *resources) {
  flagcxNetIb.deregMr(resources->netRecvComm, resources->mhandles[0]);
  flagcxNetIb.closeRecv(resources->netRecvComm);
  flagcxNetIb.closeListen(resources->netListenComm);
  deviceAdaptor->gdrMemFree(resources->buffers[0], NULL);
  for (int s = 0; s < MAXSTEPS; s++) {
    deviceAdaptor->eventDestroy(resources->cpEvents[s]);
  }
  deviceAdaptor->streamDestroy(resources->cpStream);
  return flagcxSuccess;
}
