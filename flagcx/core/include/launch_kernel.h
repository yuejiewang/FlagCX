#ifndef FLAGCX_LAUNCH_KERNEL_H_
#define FLAGCX_LAUNCH_KERNEL_H_
#pragma once
#include "adaptor.h"
#include "check.h"
#include "debug.h"
#include "flagcx.h"
#include "param.h"
#include "topo.h"
#include "utils.h"
#include <dlfcn.h>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <memory.h>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unistd.h>

struct flagcxProxyArgs;

struct flagcxSemaphore {
  flagcxSemaphore() = default;
  virtual ~flagcxSemaphore() = default;

  virtual flagcxEvent_t getEvent() = 0;
  virtual void signalStart() = 0;
  virtual void *getSignals() = 0;
  virtual void subCounter(int opId = 0) = 0;
  virtual void addCounter(int opId = 0) = 0;
  virtual int getCounter() = 0;
  virtual int pollStart(int opId = 0, int step = 0) = 0;
  virtual int pollEnd() = 0;
  virtual void wait() = 0;
  virtual bool usesStreamValue() const { return false; }
};

#define FLAGCX_OPS_PER_SEMAPHORE 64
#define FLAGCX_SIGNALS_PER_SEMAPHORE (2 * FLAGCX_OPS_PER_SEMAPHORE + 1)
#define FLAGCX_SIGNAL_CURSTEP_OFFSET 0
#define FLAGCX_SIGNAL_NSTEPS_OFFSET FLAGCX_OPS_PER_SEMAPHORE
#define FLAGCX_SIGNAL_COUNTER_OFFSET (2 * FLAGCX_OPS_PER_SEMAPHORE)

// Host semaphore derived class
struct flagcxHostSemaphore : public flagcxSemaphore {
  int counter;                              // total ops
  std::unordered_map<int, int> stepInfo;    // opId -> sigalId
  std::vector<std::pair<int, int>> signals; // [curStep, nSteps]
  std::vector<flagcxEvent_t> events;
  bool frozen; // true during execution phase

  flagcxHostSemaphore() {
    counter = 0;
    frozen = false;
    stepInfo.reserve(FLAGCX_OPS_PER_SEMAPHORE);
    signals.reserve(FLAGCX_SIGNALS_PER_SEMAPHORE);
    events.reserve(FLAGCX_SIGNALS_PER_SEMAPHORE);
  }
  ~flagcxHostSemaphore() override {
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    return event;
  }
  void signalStart() override {
    frozen =
        true; // freeze: no more structural mutations until wait() completes
    for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
      __atomic_store_n(&signals[it->second].first, 0, __ATOMIC_RELEASE);
    }
  }
  void *getSignals() override { return nullptr; }
  void subCounter(int opId = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    int idx = it->second;
    __atomic_fetch_add(&signals[idx].first, 1, __ATOMIC_RELEASE);
    INFO(FLAGCX_PROXY,
         "SubCounter curStep[%d] = %d, nSteps[%d] = %d, counter %d", opId,
         signals[idx].first, opId, signals[idx].second, counter);
  }
  void addCounter(int opId = 0) override {
    assert(!frozen); // must not mutate during execution phase
    auto it = stepInfo.find(opId);
    if (it != stepInfo.end()) {
      __atomic_fetch_add(&signals[it->second].second, 1, __ATOMIC_RELEASE);
    } else {
      signals.emplace_back(-1, 1);
      stepInfo[opId] = (int)signals.size() - 1;
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override { return counter; }
  int pollStart(int opId = 0, int step = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    return (signals[it->second].first >= step);
  }
  int pollEnd() override {
    return (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0);
  }
  void wait() override {
    int nDone = 0;
    int nOps = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
    while (nDone < nOps) {
      for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
        if (__atomic_load_n(&signals[it->second].first, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&signals[it->second].second, __ATOMIC_ACQUIRE)) {
          __atomic_fetch_add(&signals[it->second].first, 1, __ATOMIC_RELEASE);
          nDone++;
        }
      }
      sched_yield();
    }
    __atomic_store_n(&counter, 0, __ATOMIC_RELEASE);
    frozen = false; // unfreeze: allow addCounter for next round
  }
};

// Used for flagcxDeviceSemaphore to manage a buffer pool
struct flagcxDeviceSemaphoreBufferPool {
  int capacity;          // total slots
  int slotId;            // slot index in the pool
  int *signalsPool;      // Host-mapped memory region
  void *dSignalsPool;    // Device alias
  flagcxEvent_t *events; // store first event of each semaphore

  flagcxDeviceSemaphoreBufferPool();
  ~flagcxDeviceSemaphoreBufferPool();
  int getSlotId();
  void initialize();
  void setEvent(int id, flagcxEvent_t event);
  int *getHostPtr(int id);
  void *getDevicePtr(int id);
};
static flagcxDeviceSemaphoreBufferPool deviceSemaphoreBufferPool;

// Device semaphore derived class
struct flagcxDeviceSemaphore : public flagcxSemaphore {
  int slotId;
  int opOffset;
  int *signals; // [curStep,...,nSteps,..., counter]
  void *dSignals;
  flagcxEvent_t headEvent;
  std::map<int, int> curStep; // current step of each op
  std::map<int, int> nSteps;  // total steps of each op
  std::vector<flagcxEvent_t> events;

  flagcxDeviceSemaphore() {
    if (deviceSemaphoreBufferPool.capacity == -1) {
      deviceSemaphoreBufferPool.initialize();
    }
    opOffset = 0;
    slotId = deviceSemaphoreBufferPool.getSlotId();
    signals = deviceSemaphoreBufferPool.getHostPtr(slotId);
    dSignals = deviceSemaphoreBufferPool.getDevicePtr(slotId);
    headEvent = nullptr;
  }
  ~flagcxDeviceSemaphore() override {
    // Clear event in the pool
    deviceSemaphoreBufferPool.setEvent(slotId, nullptr);
    for (auto event : events) {
      deviceAdaptor->eventDestroy(event);
    }
  }
  flagcxEvent_t getEvent() override {
    events.push_back(nullptr);
    auto &event = events.back();
    deviceAdaptor->eventCreate(&event, flagcxEventDisableTiming);
    // Set the first event to the pool
    if (events.size() == 1) {
      headEvent = event;
      deviceSemaphoreBufferPool.setEvent(slotId, event);
    }
    return event;
  }
  // Since the device kernel handles the signaling,
  // host-side signalStart/End are intentionally no-op and not needed
  void signalStart() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int opId = 0) override {
    assert(curStep.find(opId) != curStep.end());
    assert(nSteps.find(opId) != nSteps.end());
    if (signals[curStep[opId]] + 1 == signals[nSteps[opId]]) {
      __atomic_fetch_sub(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 1,
                         __ATOMIC_RELEASE);
    } else {
      __atomic_fetch_add(signals + curStep[opId], 1, __ATOMIC_RELEASE);
    }
  }
  void addCounter(int opId = 0) override {
    if (nSteps.find(opId) != nSteps.end()) {
      __atomic_fetch_add(signals + nSteps[opId], 1, __ATOMIC_RELEASE);
    } else {
      // Make sure that opOffset is not used up
      assert(opOffset < FLAGCX_OPS_PER_SEMAPHORE);
      curStep[opId] = FLAGCX_SIGNAL_CURSTEP_OFFSET + opOffset;
      nSteps[opId] = FLAGCX_SIGNAL_NSTEPS_OFFSET + opOffset;
      opOffset++;
      __atomic_store_n(signals + curStep[opId], -1, __ATOMIC_RELEASE);
      __atomic_store_n(signals + nSteps[opId], 1, __ATOMIC_RELEASE);
      __atomic_fetch_add(signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 1,
                         __ATOMIC_RELEASE);
    }
  }
  int getCounter() override {
    return __atomic_load_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET,
                           __ATOMIC_ACQUIRE);
  }
  int pollStart(int opId = 0, int step = 0) override {
    assert(curStep.find(opId) != curStep.end());
    return (__atomic_load_n(signals + curStep[opId], __ATOMIC_ACQUIRE) >= step);
  }
  int pollEnd() override {
    return (__atomic_load_n(signals + FLAGCX_SIGNAL_COUNTER_OFFSET,
                            __ATOMIC_ACQUIRE) == 0);
  }
  // Since the device kernel handles the signaling,
  // host-side wait is intentionally no-op and not needed
  void wait() override {}
};

struct flagcxStreamValueSemaphore : public flagcxSemaphore {
  int counter;
  std::unordered_map<int, int> stepInfo;
  std::vector<std::pair<int, int>> steps; // [curStep, nSteps]
  uint64_t *signals;
  void *dSignals;
  int readyCount;
  int doneCount;
  flagcxEvent_t completionEvent;
  bool completionRecorded;

  flagcxStreamValueSemaphore() {
    counter = 0;
    signals = nullptr;
    dSignals = nullptr;
    readyCount = 0;
    doneCount = 0;
    completionEvent = nullptr;
    completionRecorded = false;
    stepInfo.reserve(FLAGCX_OPS_PER_SEMAPHORE);
    steps.reserve(FLAGCX_OPS_PER_SEMAPHORE);
  }
  ~flagcxStreamValueSemaphore() override {
    if (completionEvent != nullptr) {
      deviceAdaptor->eventDestroy(completionEvent);
    }
    if (signals != nullptr) {
      deviceAdaptor->deviceFree((void *)signals, flagcxMemHost, nullptr);
    }
  }

  flagcxResult_t prepare(int readyCount_, int doneCount_);
  flagcxResult_t enqueueReady(flagcxStream_t stream, flagcxStream_t launchStream,
                              int readyIdx);
  flagcxResult_t bindDoneRange(struct flagcxProxyArgs *args, int doneIdx);
  flagcxResult_t enqueueCompletion(flagcxStream_t launchStream);

  flagcxEvent_t getEvent() override { return completionEvent; }
  void signalStart() override {}
  void *getSignals() override { return dSignals; }
  void subCounter(int opId = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    int idx = it->second;
    int prev = __atomic_fetch_add(&steps[idx].first, 1, __ATOMIC_ACQ_REL);
    if (prev + 1 == __atomic_load_n(&steps[idx].second, __ATOMIC_ACQUIRE)) {
      __atomic_fetch_sub(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  void addCounter(int opId = 0) override {
    auto it = stepInfo.find(opId);
    if (it != stepInfo.end()) {
      __atomic_fetch_add(&steps[it->second].second, 1, __ATOMIC_RELEASE);
    } else {
      steps.emplace_back(0, 1);
      stepInfo[opId] = (int)steps.size() - 1;
      __atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE);
    }
  }
  int getCounter() override {
    return __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
  }
  int pollStart(int opId = 0, int step = 0) override {
    auto it = stepInfo.find(opId);
    assert(it != stepInfo.end());
    if (signals == nullptr)
      return 0;
    return (__atomic_load_n(signals, __ATOMIC_ACQUIRE) >= 1 &&
            __atomic_load_n(&steps[it->second].first, __ATOMIC_ACQUIRE) >=
                step);
  }
  int pollEnd() override;
  void wait() override {}
  bool usesStreamValue() const override { return true; }

private:
  static constexpr int kArmedIdx = 0;
  int readyBaseIdx() const { return 1; }
  int doneBaseIdx() const { return 1 + readyCount; }
  void *devicePtrAt(int idx) const {
    return static_cast<void *>(static_cast<char *>(dSignals) +
                               idx * sizeof(uint64_t));
  }
};

flagcxResult_t flagcxStreamValueSignalChunks(struct flagcxProxyArgs *args,
                                             flagcxStream_t stream,
                                             int completedChunks);

void cpuAsyncKernel(void *args);
extern flagcxLaunchFunc_t deviceAsyncKernel;

#endif
