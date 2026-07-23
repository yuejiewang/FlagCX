/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "group.h"
#include "adaptor.h"
#include "assert.h"
#include "debug.h"
#include "flagcx_hetero.h"
#include "launch_kernel.h"
#include "net.h"
#include "p2p.h"
#include "transport.h"
#include "type.h"
#include <pthread.h>
#include <queue>
#include <stdio.h>
#include <string.h>
#include <vector>

__thread int flagcxGroupDepth = 0;
__thread bool flagcxGroupJobAbortFlag = false;
__thread struct flagcxHeteroComm *flagcxGroupCommHead = nullptr;
__thread struct flagcxHeteroComm *flagcxGroupCommPreconnectHead = nullptr;
__thread flagcxResult_t flagcxGroupError = flagcxSuccess;
__thread struct flagcxGroupJob *flagcxGroupJobMainPtr = NULL;
__thread struct flagcxGroupJob flagcxGroupJobMain;
__thread int flagcxGroupBlocking = 1; /* default mode */
__thread struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
    flagcxAsyncJobs;
__thread struct flagcxGroupDeferredFree *flagcxGroupDeferredFreeHead = nullptr;
__thread struct flagcxGroupDeferredFree *flagcxGroupDeferredFreeTail = nullptr;
__thread int64_t flagcxGroupNextCustomOpId = INT_MIN;

FLAGCX_PARAM(P2pScheduleDisable, "P2P_SCHEDULE_DISABLE", 0);

flagcxResult_t flagcxHeteroGroupStart() {
  flagcxResult_t ret = flagcxSuccess;
  FLAGCXCHECK(flagcxGroupStartInternal());
  return ret;
}

flagcxResult_t flagcxHeteroGroupEnd() {
  flagcxResult_t ret = flagcxSuccess;
  FLAGCXCHECKGOTO(flagcxGroupEndInternal(), ret, exit);
exit:
  return ret;
}

int flagcxGroupAllocCustomOpId() {
  return static_cast<int>(flagcxGroupNextCustomOpId++);
}

flagcxResult_t flagcxGroupDeferFree(void *ptr, flagcxMemType_t type,
                                   flagcxStream_t stream) {
  if (flagcxGroupDepth <= 0 || ptr == nullptr) {
    return flagcxInvalidUsage;
  }
  struct flagcxGroupDeferredFree *deferred;
  FLAGCXCHECK(flagcxCalloc(&deferred, 1));
  deferred->ptr = ptr;
  deferred->type = type;
  deferred->stream = stream;
  flagcxResult_t deviceResult = deviceAdaptor->getDevice(&deferred->device);
  if (deviceResult != flagcxSuccess) {
    free(deferred);
    return deviceResult;
  }
  if (flagcxGroupDeferredFreeTail == nullptr) {
    flagcxGroupDeferredFreeHead = deferred;
  } else {
    flagcxGroupDeferredFreeTail->next = deferred;
  }
  flagcxGroupDeferredFreeTail = deferred;
  return flagcxSuccess;
}

static flagcxResult_t groupDrainDeferredFrees(struct flagcxGroupJob *gjob,
                                              bool allowStreamOrderedFree) {
  if (*gjob->deferredFreeHeadPtr == nullptr) {
    return flagcxSuccess;
  }
  flagcxResult_t ret = flagcxSuccess;
  const bool completionLaunched = gjob->deferredFreeCompletionLaunched;
  const bool streamOrderedFree =
      allowStreamOrderedFree && completionLaunched &&
      gjob->deferredFreeCompletionEvent != nullptr &&
      strcmp(deviceAdaptor->name, "CUDA") == 0;

  // Adaptors without a stream-ordered free primitive must wait here. This is
  // only the grouped in-place slow path; CUDA keeps GroupEnd asynchronous.
  if (!streamOrderedFree && completionLaunched) {
    flagcxResult_t syncResult = deviceAdaptor->streamSynchronize(
        gjob->deferredFreeCompletionStream);
    if (syncResult != flagcxSuccess) {
      ret = syncResult;
    }
  }

  int originalDevice = -1;
  flagcxResult_t getDeviceResult = deviceAdaptor->getDevice(&originalDevice);
  if (ret == flagcxSuccess && getDeviceResult != flagcxSuccess) {
    ret = getDeviceResult;
  }
  struct flagcxGroupDeferredFree *deferred = *gjob->deferredFreeHeadPtr;
  while (deferred != nullptr) {
    struct flagcxGroupDeferredFree *next = deferred->next;
    flagcxResult_t setDeviceResult = deviceAdaptor->setDevice(deferred->device);
    if (ret == flagcxSuccess && setDeviceResult != flagcxSuccess) {
      ret = setDeviceResult;
    }

    flagcxResult_t freeResult = flagcxSuccess;
    if (streamOrderedFree && setDeviceResult == flagcxSuccess) {
      flagcxStream_t freeStream = deferred->stream != nullptr
                                      ? deferred->stream
                                      : gjob->deferredFreeCompletionStream;
      flagcxResult_t waitResult = flagcxSuccess;
      if (freeStream != gjob->deferredFreeCompletionStream) {
        waitResult = deviceAdaptor->streamWaitEvent(
            freeStream, gjob->deferredFreeCompletionEvent);
      }
      if (waitResult == flagcxSuccess) {
        freeResult = deviceAdaptor->deviceFree(deferred->ptr, deferred->type,
                                               freeStream);
      } else {
        if (ret == flagcxSuccess) {
          ret = waitResult;
        }
        // Retire safely even if cross-stream event ordering failed.
        (void)deviceAdaptor->streamSynchronize(
            gjob->deferredFreeCompletionStream);
        freeResult = deviceAdaptor->deviceFree(deferred->ptr, deferred->type,
                                               nullptr);
      }
    } else if (setDeviceResult == flagcxSuccess) {
      freeResult = deviceAdaptor->deviceFree(deferred->ptr, deferred->type,
                                             nullptr);
    }
    if (ret == flagcxSuccess && freeResult != flagcxSuccess) {
      ret = freeResult;
    }
    free(deferred);
    deferred = next;
  }
  *gjob->deferredFreeHeadPtr = nullptr;
  *gjob->deferredFreeTailPtr = nullptr;
  if (originalDevice >= 0) {
    flagcxResult_t restoreResult = deviceAdaptor->setDevice(originalDevice);
    if (ret == flagcxSuccess && restoreResult != flagcxSuccess) {
      ret = restoreResult;
    }
  }
  return ret;
}

static flagcxResult_t
groupDestroyDeferredFreeEvent(struct flagcxGroupJob *gjob) {
  if (gjob->deferredFreeCompletionEvent == nullptr) {
    return flagcxSuccess;
  }
  flagcxResult_t ret =
      deviceAdaptor->eventDestroy(gjob->deferredFreeCompletionEvent);
  if (ret == flagcxSuccess) {
    gjob->deferredFreeCompletionEvent = nullptr;
  }
  return ret;
}

struct flagcxPreconnectJob {
  struct flagcxAsyncJob base;
  struct flagcxHeteroComm *comm;
};

flagcxResult_t flagcxPreconnectFunc(struct flagcxAsyncJob *job_) {
  struct flagcxPreconnectJob *job = (struct flagcxPreconnectJob *)job_;
  struct flagcxHeteroComm *comm = job->comm;
  if (comm->proxyState->initialized == 0) {
    FLAGCXCHECK(flagcxProxyInit(comm));
  }
  FLAGCXCHECK(flagcxTransportP2pSetup(comm, NULL, 0));
  return flagcxSuccess;
}

/**
 * TODO: add proxy block to make sure the connect is complete
 **/

void *flagcxAsyncJobMain(void *arg) {
  struct flagcxAsyncJob *job = (struct flagcxAsyncJob *)arg;
  // flagcxSetDevice(job->comm->cudaDev);
  deviceAdaptor->setDevice(job->comm->cudaDev);
  job->result = job->func(job);
  if (job->result != flagcxSuccess) {
    INFO(FLAGCX_INIT, "%s:%d -> %d [Async thread]", __FILE__, __LINE__,
         job->result);
  }
  __atomic_store_n(&job->state, flagcxGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}

static int64_t p2pScheduleDisable = flagcxParamP2pScheduleDisable();

static flagcxResult_t groupLaunch(struct flagcxAsyncJob *job_) {
  flagcxResult_t ret = flagcxSuccess;
  // bool errorJobAbortFlag = false;
  struct flagcxGroupJob *gjob = (struct flagcxGroupJob *)job_;
  struct flagcxHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;

  struct flagcxHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;

  struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;
  // volatile bool *groupAbortFlag = gjob->abortFlagPtr;

  // CustomizedSchedule has the highest priority, followed by P2PSchedule,
  // with DefaultSchedule as the fallback.
  // CustomizedSchedule: |op0{s0,s1,...,sN}|...|opN{s0,s1,...,sN}|
  // P2PSchedule: |recvOps{s0,s1,...,sN}|selfCopyOps{s0}|sendOps{s0,s1,...,sN}|
  // DefaultSchedule: |op0{s0}|op1{s0}|...|opN{s0}|
  int defaultOpId = 0;
  int defaultStep = 0;
  // Each groupLaunch we create a semaphore to track the
  // p2p ops and a stream to launch host or device func
  std::shared_ptr<flagcxSemaphore> semaphore;
  if (deviceAsyncKernel) {
    semaphore = std::make_shared<flagcxDeviceSemaphore>();
  } else {
    semaphore = std::make_shared<flagcxHostSemaphore>();
  }
  flagcxStream_t launchStream = nullptr;
  flagcxEvent_t launchEvent = nullptr;
  // temporary stored proxy ops in step order
  std::map<int, std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>>
      proxyOps;

  if (groupCommPreconnectHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommPreconnectHeadMain;
    do {
      struct flagcxPreconnectJob *job;
      FLAGCXCHECKGOTO(flagcxCalloc(&job, 1), ret, fail);
      job->base.func = flagcxPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = flagcxGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->comm = job->base.comm = comm;
      flagcxIntruQueueEnqueue(asyncJobsMain, &job->base);

      struct flagcxHeteroComm *next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  if (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueHead(asyncJobsMain);
    do {
      SYSCHECKGOTO(
          pthread_create(&job->thread, nullptr, flagcxAsyncJobMain, job), ret,
          fail);
      job = job->next;
    } while (job != nullptr);

    job = flagcxIntruQueueHead(asyncJobsMain);
    do {
      pthread_join(job->thread, nullptr);
      if (job->result != flagcxSuccess) {
        WARN("Async job failed with result %d", job->result);
        ret = job->result;
      }
      job = job->next;
    } while (job != nullptr);

    if (ret != flagcxSuccess)
      goto fail;
  }

  if (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    // post all send/recv tasks
    do {
      flagcxTasks *tasks = &comm->tasks;
      int nRanks = comm->nRanks;
      int localRanks = comm->localRanks;

      // Round 0: handle self send/recv (local copy)
      {
        int peer = comm->rank;
        std::vector<flagcxTaskP2p *> sendTasks;
        std::vector<flagcxTaskP2p *> recvTasks;
        while (!flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue))
          sendTasks.push_back(
              flagcxIntruQueueDequeue(&tasks->peers[peer].sendQueue));
        while (!flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
          recvTasks.push_back(
              flagcxIntruQueueDequeue(&tasks->peers[peer].recvQueue));

        for (size_t i = 0; i < sendTasks.size();) {
          bool matched = false;
          for (size_t j = 0; j < recvTasks.size(); j++) {
            if (sendTasks[i]->bytes == recvTasks[j]->bytes &&
                sendTasks[i]->dtype == recvTasks[j]->dtype &&
                sendTasks[i]->opId == recvTasks[j]->opId &&
                sendTasks[i]->step == recvTasks[j]->step) {
              if (sendTasks[i]->buff != recvTasks[j]->buff) {
                flagcxProxyOp *op;
                FLAGCXCHECK(flagcxCalloc(&op, 1));
                op->pattern = flagcxPatternSend;
                op->nbytes = sendTasks[i]->bytes;
                op->sendbuff = (uint8_t *)sendTasks[i]->buff;
                op->recvbuff = (uint8_t *)recvTasks[j]->buff;
                op->channelId = 0;
                op->root = peer;
                op->connection = comm->channels[op->channelId]
                                     .peers[peer]
                                     ->send[0]
                                     .proxyConn.connection;
                op->stream = sendTasks[i]->stream;
                op->event = semaphore->getEvent();
                op->args.chunkSteps = 1; // single step
                op->args.semaphore = semaphore;
                op->args.opId = sendTasks[i]->opId == INT_MAX
                                    ? (p2pScheduleDisable ? defaultOpId : 0)
                                    : sendTasks[i]->opId;
                op->args.step = sendTasks[i]->step == -1
                                    ? (p2pScheduleDisable ? defaultStep : 0)
                                    : sendTasks[i]->step;
                semaphore->addCounter(op->args.opId);
                defaultOpId++;
                FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
                if (launchStream == nullptr) {
                  launchStream = op->stream;
                  launchEvent = op->event;
                } else {
                  FLAGCXCHECK(
                      deviceAdaptor->streamWaitEvent(launchStream, op->event));
                }
                if (proxyOps.find(op->args.step) == proxyOps.end()) {
                  proxyOps[op->args.step] = std::vector<
                      std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
                }
                proxyOps[op->args.step].push_back({comm, op});
              }
              free(sendTasks[i]);
              free(recvTasks[j]);
              sendTasks.erase(sendTasks.begin() + i);
              recvTasks.erase(recvTasks.begin() + j);
              matched = true;
              break;
            }
          }
          if (!matched)
            i++;
        }
        for (auto *task : sendTasks)
          flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, task);
        for (auto *task : recvTasks)
          flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, task);
      }

      // Round 1..nRanks-1: use p2pSchedule to pair recv/send with different
      // peers
      int roundSendStep = 0;
      int roundRecvStep = 0;
      int roundOpId = 1;
      for (int round = 1; round < nRanks; round++) {
        int tmpRoundOpId = round / localRanks + 1;
        if (roundOpId != tmpRoundOpId) {
          roundSendStep = 0;
          roundRecvStep = 0;
          roundOpId = tmpRoundOpId;
        }
        int recvPeer = comm->p2pSchedule[round].recvRank;
        int sendPeer = comm->p2pSchedule[round].sendRank;
        while (!flagcxIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue) ||
               !flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
          // Process one recv task (for IPC register)
          if (!flagcxIntruQueueEmpty(&tasks->peers[recvPeer].recvQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[recvPeer].recvQueue);
            int peer = recvPeer;
            flagcxProxyOp *op;
            FLAGCXCHECK(flagcxCalloc(&op, 1));
            op->pattern = flagcxPatternRecv;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->recv[0]
                                 .proxyConn.connection;
            op->stream = p2p->stream;
            if (op->connection == NULL) {
              WARN("groupLaunch: recv proxyConn.connection is NULL for rank %d "
                   "peer %d channel %d",
                   comm->rank, peer, op->channelId);
              return flagcxInternalError;
            }
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = flagcxP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Receiver: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL("Receiver: [peerRank(%d), rank(%d)] -> "
                         "[peerSlotIdx(%ld), peerOpHash(%ld)]",
                         peer, comm->rank, op->args.p2pPeerSlotIdx,
                         op->args.p2pPeerOpHash);

              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              op->args.regBufFlag = 0;
              FLAGCXCHECK(flagcxP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerRanks, 1,
                  &op->args.regBufFlag, &regOffset, &peerRmtAddr));
              if (op->args.regBufFlag && peerRmtAddr) {
                op->args.p2pRmtAddr = (void *)peerRmtAddr;
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = flagcxNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + flagcxNetChunkSize - 1) / (flagcxNetChunkSize);
              op->args.sendStepMask = flagcxNetChunks - 1;
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->recv};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId =
                p2p->opId == INT_MAX
                    ? (p2pScheduleDisable ? defaultOpId : -roundOpId)
                    : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundRecvStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundRecvStep++;
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
          // Process one send task (for IPC lookup - after recv's register)
          if (!flagcxIntruQueueEmpty(&tasks->peers[sendPeer].sendQueue)) {
            flagcxTaskP2p *p2p =
                flagcxIntruQueueDequeue(&tasks->peers[sendPeer].sendQueue);
            int peer = sendPeer;
            flagcxProxyOp *op;
            FLAGCXCHECK(flagcxCalloc(&op, 1));
            op->pattern = flagcxPatternSend;
            op->nbytes = p2p->bytes;
            op->recvbuff = (uint8_t *)p2p->buff;
            op->channelId = 0;
            op->root = peer;
            op->connection = comm->channels[op->channelId]
                                 .peers[peer]
                                 ->send[0]
                                 .proxyConn.connection;
            op->stream = p2p->stream;
            if (op->connection == NULL) {
              WARN("groupLaunch: send proxyConn.connection is NULL for rank %d "
                   "peer %d channel %d",
                   comm->rank, peer, op->channelId);
              return flagcxInternalError;
            }
            if (op->connection->transport == TRANSPORT_P2P) {
              op->args.chunkSize = computeP2pChunkSize(p2p->bytes);
              op->args.chunkSteps =
                  (p2p->bytes + op->args.chunkSize - 1) / (op->args.chunkSize);
              op->args.sendStepMask = flagcxP2pChunks - 1;
              setP2pSlotInfo(comm->rank, peer, p2p->bytes, p2p->dtype, 0,
                             &op->args.p2pOpHash, &op->args.p2pSlotIdx);
              setP2pSlotInfo(peer, comm->rank, p2p->bytes, p2p->dtype, 1,
                             &op->args.p2pPeerOpHash, &op->args.p2pPeerSlotIdx);
              TRACE_CALL("Sender: [rank(%d), peerRank(%d)] -> [slotIdx(%ld), "
                         "opHash(%ld)]",
                         comm->rank, peer, op->args.p2pSlotIdx,
                         op->args.p2pOpHash);
              TRACE_CALL(
                  "Sender: [peerRank(%d), rank(%d)] -> [peerSlotIdx(%ld), "
                  "peerOpHash(%ld)]",
                  peer, comm->rank, op->args.p2pPeerSlotIdx,
                  op->args.p2pPeerOpHash);
              // Send side: register own buffer to peer's proxy for READ mode.
              // The actual IPC address comes from SHM at proxy time.
              int peerRanks[] = {peer};
              uintptr_t regOffset = 0;
              uintptr_t *peerRmtAddr = NULL;
              op->args.regBufFlag = 0;
              FLAGCXCHECK(flagcxP2pRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerRanks, 1,
                  &op->args.regBufFlag, &regOffset, &peerRmtAddr));
              if (op->args.regBufFlag && peerRmtAddr) {
                op->args.p2pRmtAddr = (void *)peerRmtAddr;
              }
            } else if (op->connection->transport == TRANSPORT_NET) {
              op->args.chunkSize = flagcxNetChunkSize;
              op->args.chunkSteps =
                  (p2p->bytes + flagcxNetChunkSize - 1) / (flagcxNetChunkSize);
              op->args.sendStepMask = flagcxNetChunks - 1;
              flagcxConnector *peerConns[] = {
                  comm->channels[op->channelId].peers[peer]->send};
              FLAGCXCHECK(flagcxNetRegisterBuffer(
                  comm, p2p->buff, p2p->bytes, peerConns, 1,
                  &op->args.regBufFlag, &op->args.regHandle));
            }
            op->args.semaphore = semaphore;
            op->args.opId = p2p->opId == INT_MAX
                                ? (p2pScheduleDisable ? defaultOpId : roundOpId)
                                : p2p->opId;
            op->args.step =
                p2p->step == -1
                    ? (p2pScheduleDisable ? defaultStep : roundSendStep)
                    : p2p->step;
            op->event = semaphore->getEvent();
            semaphore->addCounter(op->args.opId);
            defaultOpId++;
            roundSendStep++;
            FLAGCXCHECK(deviceAdaptor->eventRecord(op->event, op->stream));
            if (launchStream == nullptr) {
              launchStream = op->stream;
              launchEvent = op->event;
            } else {
              FLAGCXCHECK(
                  deviceAdaptor->streamWaitEvent(launchStream, op->event));
            }
            if (proxyOps.find(op->args.step) == proxyOps.end()) {
              proxyOps[op->args.step] =
                  std::vector<std::pair<flagcxHeteroComm *, flagcxProxyOp *>>();
            }
            proxyOps[op->args.step].push_back({comm, op});
            free(p2p);
          }
        }
      }
      tasks->p2pOrderSteps = 0;
      comm = comm->groupNext;
    } while (comm != nullptr);
  }

  // Save all proxy ops in step order
  for (auto it = proxyOps.begin(); it != proxyOps.end(); ++it) {
    for (auto pair : it->second) {
      FLAGCXCHECK(flagcxProxySaveOp(pair.first, pair.second));
    }
  }

  if (*gjob->deferredFreeHeadPtr != nullptr && launchStream != nullptr) {
    FLAGCXCHECKGOTO(deviceAdaptor->eventCreate(
                        &gjob->deferredFreeCompletionEvent,
                        flagcxEventDisableTiming),
                    ret, fail);
  }

  if (launchStream != nullptr && launchEvent != nullptr) {
    if (deviceAsyncKernel) {
      FLAGCXCHECK(deviceAdaptor->launchDeviceFunc(
          launchStream, deviceAsyncKernel, (void *)semaphore->getSignals()));
      if (gjob->deferredFreeCompletionEvent != nullptr) {
        gjob->deferredFreeCompletionStream = launchStream;
        gjob->deferredFreeCompletionLaunched = true;
      }
      // device semaphore need this event to signal completion
      FLAGCXCHECK(deviceAdaptor->eventRecord(launchEvent, launchStream));
    } else {
      FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
      if (gjob->deferredFreeCompletionEvent != nullptr) {
        gjob->deferredFreeCompletionStream = launchStream;
        gjob->deferredFreeCompletionLaunched = true;
      }
    }
  }

  if (gjob->deferredFreeCompletionEvent != nullptr) {
    FLAGCXCHECKGOTO(deviceAdaptor->eventRecord(
                        gjob->deferredFreeCompletionEvent, launchStream),
                    ret, fail);
  }

  // Free group-owned temporary buffers only after the completion callback (or
  // device semaphore kernel) above. CUDA queues a wait/free on each allocation
  // stream; adaptors without stream-ordered free wait before releasing.
  FLAGCXCHECKGOTO(groupDrainDeferredFrees(gjob, true), ret, fail);
  FLAGCXCHECKGOTO(groupDestroyDeferredFreeEvent(gjob), ret, fail);

  while (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  while (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    struct flagcxHeteroComm *next = comm->groupNext;
    (void)flagcxGroupCommLeave(comm);
    groupCommHeadMain = next;
  }
exit:
  return ret;
fail:
  goto exit;
}

static flagcxResult_t groupCleanup(struct flagcxAsyncJob *job_) {
  struct flagcxGroupJob *gjob = (struct flagcxGroupJob *)job_;
  struct flagcxHeteroComm *groupCommHeadMain = *gjob->groupCommHeadPtr;
  struct flagcxHeteroComm *groupCommPreconnectHeadMain =
      *gjob->groupCommPreconnectHeadPtr;
  struct flagcxIntruQueue<struct flagcxAsyncJob, &flagcxAsyncJob::next>
      *asyncJobsMain = gjob->asyncJobsPtr;

  // clean up preconnect comms
  while (groupCommPreconnectHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommPreconnectHeadMain;
    struct flagcxHeteroComm *next = comm->preconnectNext;
    comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
    groupCommPreconnectHeadMain = next;
  }

  // clean up async jobs
  while (!flagcxIntruQueueEmpty(asyncJobsMain)) {
    struct flagcxAsyncJob *job = flagcxIntruQueueDequeue(asyncJobsMain);
    free(job);
  }

  // clean up comms
  while (groupCommHeadMain != nullptr) {
    struct flagcxHeteroComm *comm = groupCommHeadMain;
    struct flagcxHeteroComm *next = comm->groupNext;
    (void)flagcxGroupCommLeave(comm);
    groupCommHeadMain = next;
  }

  flagcxResult_t deferredFreeResult = groupDrainDeferredFrees(gjob, false);
  flagcxResult_t eventResult = groupDestroyDeferredFreeEvent(gjob);
  return deferredFreeResult != flagcxSuccess ? deferredFreeResult : eventResult;
}

static inline void groupResetJobState() {
  flagcxGroupBlocking = 0;
  flagcxGroupJobMainPtr = NULL;
  flagcxGroupCommPreconnectHead = nullptr;
  flagcxGroupCommHead = nullptr;
  flagcxGroupDeferredFreeHead = nullptr;
  flagcxGroupDeferredFreeTail = nullptr;
  memset(&flagcxGroupJobMain, 0, sizeof(struct flagcxGroupJob));
}

flagcxResult_t flagcxGroupEndInternal() {
  flagcxResult_t ret = flagcxSuccess;
  flagcxGroupDepth--;
  if (flagcxGroupDepth < 0)
    return flagcxSystemError;
  if (flagcxGroupDepth == 0) {
    if (flagcxGroupCommPreconnectHead || flagcxGroupCommHead ||
        flagcxGroupDeferredFreeHead) {
      flagcxGroupJobMain.groupCommHeadPtr = &flagcxGroupCommHead;
      flagcxGroupJobMain.groupCommPreconnectHeadPtr =
          &flagcxGroupCommPreconnectHead;
      flagcxGroupJobMain.asyncJobsPtr = &flagcxAsyncJobs;
      flagcxGroupJobMain.deferredFreeHeadPtr = &flagcxGroupDeferredFreeHead;
      flagcxGroupJobMain.deferredFreeTailPtr = &flagcxGroupDeferredFreeTail;
      flagcxGroupJobMain.initialized = true;
      flagcxGroupJobMainPtr = &flagcxGroupJobMain;
      FLAGCXCHECKGOTO(groupLaunch(&flagcxGroupJobMainPtr->base), ret, fail);
      groupResetJobState();
    }
  }

exit:
  return ret;
fail:
  groupCleanup(&flagcxGroupJobMainPtr->base);
  groupResetJobState();
  goto exit;
}
