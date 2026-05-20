#include "launch_kernel.h"
#include "group.h"
#include "proxy.h"
#include <algorithm>
#include <stdio.h>

flagcxLaunchFunc_t deviceAsyncKernel = NULL;

FLAGCX_PARAM(SemaphoreBufferPoolCapacity, "SEMAPHORE_BUFFER_POOL_CAPACITY", 32);

flagcxDeviceSemaphoreBufferPool::flagcxDeviceSemaphoreBufferPool()
    : capacity(-1), slotId(-1), signalsPool(nullptr), dSignalsPool(nullptr),
      events(nullptr) {}

flagcxDeviceSemaphoreBufferPool::~flagcxDeviceSemaphoreBufferPool() {
  free(events);
  dSignalsPool = nullptr;
  if (signalsPool != nullptr) {
    deviceAdaptor->deviceFree((void *)signalsPool, flagcxMemHost, nullptr);
  }
}

int flagcxDeviceSemaphoreBufferPool::getSlotId() {
  assert(capacity != -1);
  if (events[slotId] != nullptr) {
    // wait for the previous event to complete
    while (deviceAdaptor->eventQuery(events[slotId]) != flagcxSuccess) {
      sched_yield();
    }
    events[slotId] = nullptr;
  }
  // set this slot signals to zero
  int offset = FLAGCX_SIGNALS_PER_SEMAPHORE * slotId;
  memset(signalsPool + offset, 0, FLAGCX_SIGNALS_PER_SEMAPHORE * sizeof(int));
  int ret = slotId;
  // Move to next slot
  slotId = (slotId + 1) % capacity;
  return ret;
}

void flagcxDeviceSemaphoreBufferPool::initialize() {
  capacity = flagcxParamSemaphoreBufferPoolCapacity();
  slotId = 0;
  // Allocate host-pinned memory for all semaphores (3 ints each)
  deviceAdaptor->deviceMalloc((void **)&signalsPool,
                              capacity * FLAGCX_SIGNALS_PER_SEMAPHORE *
                                  sizeof(int),
                              flagcxMemHost, nullptr);
  // Get device pointer alias
  deviceAdaptor->hostGetDevicePointer(&dSignalsPool, (void *)signalsPool);
  // Init events to nullptr
  flagcxCalloc(&events, capacity);
  for (int i = 0; i < capacity; i++) {
    events[i] = nullptr;
  }
}

// Set event for a semaphore
void flagcxDeviceSemaphoreBufferPool::setEvent(int id, flagcxEvent_t event) {
  assert(id >= 0 && id < capacity);
  // events[id] should be set to nullptr before
  events[id] = event;
}

// Return pointer to the start of a semaphore’s signals (host/device)
int *flagcxDeviceSemaphoreBufferPool::getHostPtr(int id) {
  assert(id >= 0 && id < capacity);
  return signalsPool + FLAGCX_SIGNALS_PER_SEMAPHORE * id;
}
void *flagcxDeviceSemaphoreBufferPool::getDevicePtr(int id) {
  assert(id >= 0 && id < capacity);
  return static_cast<void *>((static_cast<char *>(dSignalsPool) +
                              FLAGCX_SIGNALS_PER_SEMAPHORE * id * sizeof(int)));
}

void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  semaphore->wait();
}

flagcxResult_t flagcxStreamValueSemaphore::prepare(int readyCount_,
                                                   int doneCount_) {
  readyCount = readyCount_;
  doneCount = doneCount_;
  size_t signalCount = 1 + readyCount + doneCount;

  FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&signals,
                                          signalCount * sizeof(uint64_t),
                                          flagcxMemHost, nullptr));
  memset(signals, 0, signalCount * sizeof(uint64_t));
  FLAGCXCHECK(deviceAdaptor->hostGetDevicePointer(&dSignals, (void *)signals));
  FLAGCXCHECK(
      deviceAdaptor->eventCreate(&completionEvent, flagcxEventDisableTiming));
  completionRecorded = false;
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueSemaphore::enqueueReady(flagcxStream_t stream,
                                                        flagcxStream_t launchStream,
                                                        int readyIdx) {
  if (stream == nullptr || launchStream == nullptr) {
    WARN("stream-value group requires non-null streams");
    return flagcxInvalidUsage;
  }
  if (readyIdx < 0 || readyIdx >= readyCount) {
    WARN("readyIdx %d out of range %d", readyIdx, readyCount);
    return flagcxInternalError;
  }
  void *readyPtr = devicePtrAt(readyBaseIdx() + readyIdx);
  FLAGCXCHECK(deviceAdaptor->streamWriteValue64(stream, readyPtr, 1, 0));
  FLAGCXCHECK(
      deviceAdaptor->streamWaitValue64(launchStream, readyPtr, 1, 0));
  return flagcxSuccess;
}

flagcxResult_t flagcxStreamValueSemaphore::bindDoneRange(
    struct flagcxProxyArgs *args, int doneIdx) {
  if (args == nullptr)
    return flagcxInvalidArgument;
  args->streamValueDoneCount = args->chunkSteps;
  args->streamValueDoneWritten = 0;
  if (args->chunkSteps == 0) {
    args->streamValueDonePtr = nullptr;
    return flagcxSuccess;
  }
  if (doneIdx < 0 || doneIdx + args->chunkSteps > doneCount) {
    WARN("done flag range [%d, %d) exceeds %d", doneIdx,
         doneIdx + args->chunkSteps, doneCount);
    return flagcxInternalError;
  }
  args->streamValueDonePtr = devicePtrAt(doneBaseIdx() + doneIdx);
  return flagcxSuccess;
}

flagcxResult_t
flagcxStreamValueSemaphore::enqueueCompletion(flagcxStream_t launchStream) {
  if (launchStream == nullptr) {
    WARN("stream-value group requires a launch stream");
    return flagcxInvalidUsage;
  }
  FLAGCXCHECK(deviceAdaptor->streamWriteValue64(
      launchStream, devicePtrAt(kArmedIdx), 1, 0));
  for (int i = 0; i < doneCount; ++i) {
    FLAGCXCHECK(deviceAdaptor->streamWaitValue64(
        launchStream, devicePtrAt(doneBaseIdx() + i), 1, 0));
  }
  FLAGCXCHECK(deviceAdaptor->eventRecord(completionEvent, launchStream));
  completionRecorded = true;
  return flagcxSuccess;
}

int flagcxStreamValueSemaphore::pollEnd() {
  if (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) != 0)
    return 0;
  if (!completionRecorded || completionEvent == nullptr)
    return 0;
  return deviceAdaptor->eventQuery(completionEvent) == flagcxSuccess;
}

flagcxResult_t flagcxStreamValueSignalChunks(struct flagcxProxyArgs *args,
                                             flagcxStream_t stream,
                                             int completedChunks) {
  if (args == nullptr || args->streamValueDonePtr == nullptr ||
      args->streamValueDoneCount == 0) {
    return flagcxSuccess;
  }
  if (stream == nullptr) {
    WARN("stream-value chunk signaling requires a non-null stream");
    return flagcxInvalidUsage;
  }

  int target = std::min(completedChunks, args->streamValueDoneCount);
  while (args->streamValueDoneWritten < target) {
    void *donePtr = static_cast<void *>(
        static_cast<char *>(args->streamValueDonePtr) +
        args->streamValueDoneWritten * sizeof(uint64_t));
    FLAGCXCHECK(deviceAdaptor->streamWriteValue64(stream, donePtr, 1, 0));
    args->streamValueDoneWritten++;
  }
  return flagcxSuccess;
}
