#include "ascend_adaptor.h"

#ifdef USE_ASCEND_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "driver/ascend_hal.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <utility>

std::map<flagcxMemcpyType_t, aclrtMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, ACL_MEMCPY_HOST_TO_DEVICE},
    {flagcxMemcpyDeviceToHost, ACL_MEMCPY_DEVICE_TO_HOST},
    {flagcxMemcpyDeviceToDevice, ACL_MEMCPY_DEVICE_TO_DEVICE},
};

namespace {

using CannIpcKey =
    std::array<char, FLAGCX_ASCEND_IPC_KEY_BUFFER_BYTES>;
using CannDevicePtrKey = std::pair<int, uintptr_t>;
using CannImportKey = std::pair<int, std::string>;

struct CannImportedIpcRecord {
  CannIpcKey key;
  size_t refCount;
};

struct CannExportedIpcRecord {
  CannIpcKey key;
  size_t refCount;
};

struct CannHostRegistration {
  size_t size;
  size_t refCount;
};

static_assert(sizeof(flagcxIpcMemHandle) ==
                  FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES,
              "FlagCX must transport exactly the 64 non-NUL ACL IPC key "
              "bytes");

std::mutex cannIpcMutex;
std::map<CannDevicePtrKey, CannExportedIpcRecord> cannExportedIpcKeys;
std::map<CannDevicePtrKey, CannImportedIpcRecord> cannImportedIpcPtrs;
std::map<CannImportKey, void *> cannImportedIpcKeys;
std::map<CannDevicePtrKey, size_t> cannDeviceAllocationSizes;

std::mutex cannHostRegistrationMutex;
std::map<uintptr_t, CannHostRegistration> cannHostRegistrations;

flagcxResult_t cannGetCurrentDevice(int *device) {
  if (device == nullptr)
    return flagcxInvalidArgument;
  aclError ret = aclrtGetDevice(device);
  return ret == ACL_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}

CannIpcKey cannRebuildIpcKey(flagcxIpcMemHandle_t handle) {
  CannIpcKey key = {};
  memcpy(key.data(), handle->key, FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES);
  key[FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES] = '\0';
  return key;
}

std::string cannSerializedIpcKey(const CannIpcKey &key) {
  return std::string(key.data(), FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES);
}

flagcxResult_t cannRegisterHostMemory(void *ptr, size_t size, uint32_t flags) {
  if (ptr == nullptr || size == 0)
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lock(cannHostRegistrationMutex);
  auto it = cannHostRegistrations.find(reinterpret_cast<uintptr_t>(ptr));
  if (it != cannHostRegistrations.end()) {
    if (it->second.size != size)
      return flagcxInvalidArgument;
    it->second.refCount++;
    return flagcxSuccess;
  }

  aclError ret = aclrtHostRegisterV2(ptr, size, flags);
  if (ret != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  cannHostRegistrations.emplace(
      reinterpret_cast<uintptr_t>(ptr),
      CannHostRegistration{size, static_cast<size_t>(1)});
  return flagcxSuccess;
}

flagcxResult_t cannUnregisterHostMemory(void *ptr) {
  if (ptr == nullptr)
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lock(cannHostRegistrationMutex);
  auto it = cannHostRegistrations.find(reinterpret_cast<uintptr_t>(ptr));
  if (it == cannHostRegistrations.end())
    return flagcxInvalidArgument;
  if (it->second.refCount > 1) {
    it->second.refCount--;
    return flagcxSuccess;
  }

  aclError ret = aclrtHostUnregister(ptr);
  if (ret != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  cannHostRegistrations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t cannReleaseExportedIpcMemory(void *devPtr,
                                            bool freeingAllocation) {
  int device = 0;
  flagcxResult_t res = cannGetCurrentDevice(&device);
  if (res != flagcxSuccess)
    return res;

  std::lock_guard<std::mutex> lock(cannIpcMutex);
  auto it = cannExportedIpcKeys.find(
      CannDevicePtrKey{device, reinterpret_cast<uintptr_t>(devPtr)});
  if (it == cannExportedIpcKeys.end())
    return flagcxSuccess;

  // Local exporter refcounts do not account for mappings imported by other
  // processes. A raw allocation free therefore cannot prove ACL's required
  // importer-before-exporter ordering. Ordered owners (for example Ascend
  // UniRunner cleanup) must explicitly close the exporter first; only then
  // may deviceFree release the allocation.
  if (freeingAllocation)
    return flagcxInvalidUsage;
  if (it->second.refCount > 1) {
    it->second.refCount--;
    return flagcxSuccess;
  }

  aclError ret = aclrtIpcMemClose(it->second.key.data());
  if (ret != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  cannExportedIpcKeys.erase(it);
  return flagcxSuccess;
}

} // namespace

flagcxResult_t cannAdaptorDeviceSynchronize() {
  DEVCHECK(aclrtSynchronizeDevice());
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(aclrtMemcpy(dst, size, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(aclrtMemcpyAsync(dst, size, src, size, memcpy_type_map[type],
                              stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(aclrtMemset(ptr, size, value, size));
    } else {
      DEVCHECK(aclrtMemsetAsync(ptr, size, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  (void)stream;
  if (ptr == nullptr || size == 0)
    return flagcxInvalidArgument;

  if (type == flagcxMemHost) {
    DEVCHECK(aclrtMallocHost(ptr, size));
  } else {
    int device = 0;
    FLAGCXCHECK(cannGetCurrentDevice(&device));
    aclError ret = aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS)
      return flagcxUnhandledDeviceError;
    std::lock_guard<std::mutex> lock(cannIpcMutex);
    cannDeviceAllocationSizes[CannDevicePtrKey{
        device, reinterpret_cast<uintptr_t>(*ptr)}] = size;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  (void)stream;
  if (ptr == nullptr)
    return flagcxSuccess;

  if (type == flagcxMemHost) {
    DEVCHECK(aclrtFreeHost(ptr));
  } else {
    int device = 0;
    FLAGCXCHECK(cannGetCurrentDevice(&device));
    // ACL requires the exporting process to close its IPC key before freeing
    // the allocation. Imported mappings are closed through ipcMemHandleClose.
    FLAGCXCHECK(cannReleaseExportedIpcMemory(ptr, true));
    aclError ret = aclrtFree(ptr);
    if (ret != ACL_SUCCESS)
      return flagcxUnhandledDeviceError;
    std::lock_guard<std::mutex> lock(cannIpcMutex);
    cannDeviceAllocationSizes.erase(
        CannDevicePtrKey{device, reinterpret_cast<uintptr_t>(ptr)});
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorSetDevice(int dev) {
  DEVCHECK(aclrtSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDevice(int *dev) {
  DEVCHECK(aclrtGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDeviceCount(int *count) {
  DEVCHECK(aclrtGetDeviceCount((uint32_t *)count));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ASCEND");
  return flagcxSuccess;
}

static flagcxResult_t cannAdaptorGetPciBdf(int dev, uint64_t *bdf) {
  if (dev < 0 || bdf == nullptr)
    return flagcxInvalidArgument;

  int64_t value = 0;
  const drvError_t result =
      halGetDeviceInfo(static_cast<uint32_t>(dev), MODULE_TYPE_PCIE,
                       INFO_TYPE_ID, &value);
  if (result != 0 || value < 0) {
    WARN("CANN halGetDeviceInfo PCI BDF failed for logical device %d: "
         "result=%d value=%ld",
         dev, static_cast<int>(result), static_cast<long>(value));
    return flagcxUnhandledDeviceError;
  }
  *bdf = static_cast<uint64_t>(value);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == nullptr || len <= 0)
    return flagcxInvalidArgument;

  uint64_t bdf = 0;
  FLAGCXCHECK(cannAdaptorGetPciBdf(dev, &bdf));
  const unsigned int domain = static_cast<unsigned int>((bdf >> 16) & 0xffff);
  const unsigned int bus = static_cast<unsigned int>((bdf >> 8) & 0xff);
  const unsigned int device = static_cast<unsigned int>((bdf >> 3) & 0x1f);
  const unsigned int function = static_cast<unsigned int>(bdf & 0x7);
  const int written = snprintf(pciBusId, static_cast<size_t>(len),
                               "%04x:%02x:%02x.%01x", domain, bus, device,
                               function);
  return written >= 0 && written < len ? flagcxSuccess
                                      : flagcxInvalidArgument;
}

flagcxResult_t cannAdaptorGetDeviceByPciBusId(int *dev,
                                              const char *pciBusId) {
  if (dev == nullptr || pciBusId == nullptr)
    return flagcxInvalidArgument;

  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  int consumed = 0;
  if (sscanf(pciBusId, "%x:%x:%x.%x%n", &domain, &bus, &device, &function,
             &consumed) != 4 ||
      pciBusId[consumed] != '\0' ||
      domain > 0xffff || bus > 0xff || device > 0x1f || function > 0x7)
    return flagcxInvalidArgument;
  const uint64_t expected = (static_cast<uint64_t>(domain) << 16) |
                            (static_cast<uint64_t>(bus) << 8) |
                            (static_cast<uint64_t>(device) << 3) | function;

  int count = 0;
  FLAGCXCHECK(cannAdaptorGetDeviceCount(&count));
  for (int candidate = 0; candidate < count; ++candidate) {
    uint64_t candidateBdf = 0;
    flagcxResult_t result =
        cannAdaptorGetPciBdf(candidate, &candidateBdf);
    if (result != flagcxSuccess)
      return result;
    if (candidateBdf == expected) {
      *dev = candidate;
      return flagcxSuccess;
    }
  }
  return flagcxInvalidArgument;
}

flagcxResult_t cannAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == nullptr || pHost == nullptr)
    return flagcxInvalidArgument;
  *pDevice = nullptr;
  DEVCHECK(aclrtHostGetDevicePointer(pHost, pDevice, 0));
  return *pDevice != nullptr ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// TODO:unsupport
flagcxResult_t cannAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
  return flagcxSuccess;
}

// TODO:unsupported
flagcxResult_t cannAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(aclrtFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCreate(flagcxStream_t *stream) {
  if (stream == nullptr)
    return flagcxInvalidArgument;
  (*stream) = NULL;
  FLAGCXCHECK(flagcxCalloc(stream, 1));
  aclError ret = aclrtCreateStream(&((*stream)->base));
  if (ret != ACL_SUCCESS) {
    free(*stream);
    *stream = NULL;
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtDestroyStream(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (aclrtStream)oldStream;
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(aclrtSynchronizeStream(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    aclrtStreamStatus status;
    DEVCHECK(aclrtStreamQuery(stream->base, &status));
    if (status == ACL_STREAM_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_STREAM_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t cannAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(aclrtStreamWaitEvent(stream->base, event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  if (event == nullptr)
    return flagcxInvalidArgument;
  (*event) = NULL;
  FLAGCXCHECK(flagcxCalloc(event, 1));
  const unsigned int flags =
      (eventType == flagcxEventDefault) ? ACL_EVENT_TIME_LINE : ACL_EVENT_SYNC;
  aclError ret = aclrtCreateEventWithFlag(&((*event)->base), flags);
  if (ret != ACL_SUCCESS) {
    free(*event);
    *event = NULL;
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtDestroyEvent(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(aclrtRecordEvent(event->base, stream->base));
    } else {
      return flagcxUnhandledDeviceError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(aclrtSynchronizeEvent(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    // cpEvents are recorded on the NET copy stream and polled by the proxy.
    // Query the completion of the tasks captured by aclrtRecordEvent; the
    // WaitStatus API instead reports a separate aclrtStreamWaitEvent task.
    aclrtEventRecordedStatus status;
    DEVCHECK(aclrtQueryEventStatus(event->base, &status));
    if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) {
      res = flagcxSuccess;
    } else if (status == ACL_EVENT_RECORDED_STATUS_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t cannAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  if (handle == nullptr)
    return flagcxInvalidArgument;
  *handle = nullptr;
  FLAGCXCHECK(flagcxCalloc(handle, 1));
  if (size != nullptr)
    *size = sizeof(flagcxIpcMemHandle);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemGetProcessId(int32_t *processId) {
  if (processId == nullptr)
    return flagcxInvalidArgument;
  aclError ret = aclrtDeviceGetBareTgid(processId);
  if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemGetProcessId: aclrtDeviceGetBareTgid failed "
         "ret=%d",
         static_cast<int>(ret));
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleSetImportPid(
    flagcxIpcMemHandle_t handle, int32_t *processIds, size_t count) {
  if (handle == nullptr || processIds == nullptr || count == 0)
    return flagcxInvalidArgument;

  CannIpcKey key = cannRebuildIpcKey(handle);
  if (key[0] == '\0')
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lock(cannIpcMutex);
  aclError ret =
      aclrtIpcMemSetImportPid(key.data(), processIds, count);
  if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemHandleSetImportPid: "
         "aclrtIpcMemSetImportPid failed ret=%d count=%zu",
         static_cast<int>(ret), count);
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemEnablePeerAccess(int peerDevice) {
  if (peerDevice < 0)
    return flagcxInvalidArgument;
  aclError ret = aclrtDeviceEnablePeerAccess(peerDevice, 0);
  if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemEnablePeerAccess: "
         "aclrtDeviceEnablePeerAccess failed ret=%d peerDevice=%d",
         static_cast<int>(ret), peerDevice);
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == nullptr || devPtr == nullptr)
    return flagcxInvalidArgument;
  memset(handle->key, 0, FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES);

  int device = 0;
  FLAGCXCHECK(cannGetCurrentDevice(&device));
  void *allocationBase = nullptr;
  size_t allocationSize = 0;
  aclError ret =
      aclrtMemGetAddressRange(devPtr, &allocationBase, &allocationSize);
  if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
    // CANN 9.0.0 beta runtimes expose aclrtMemGetAddressRange in their ABI but
    // return FEATURE_NOT_SUPPORT on Atlas A2/910B. Recover the exact allocation
    // extent recorded by this adaptor's allocator and use the documented
    // direct aclrtMalloc -> aclrtIpcMemGetExportKey flow.
    std::lock_guard<std::mutex> lock(cannIpcMutex);
    auto allocation = cannDeviceAllocationSizes.find(
        CannDevicePtrKey{device, reinterpret_cast<uintptr_t>(devPtr)});
    if (allocation == cannDeviceAllocationSizes.end()) {
      WARN("cannAdaptorIpcMemHandleGet: aclrtMemGetAddressRange is unsupported "
           "and allocation size is unknown device=%d devPtr=%p",
           device, devPtr);
      return flagcxNotSupported;
    }
    allocationBase = devPtr;
    allocationSize = allocation->second;
  } else if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemHandleGet: aclrtMemGetAddressRange failed "
         "ret=%d devPtr=%p",
         static_cast<int>(ret), devPtr);
    return flagcxUnhandledDeviceError;
  }
  // An ACL IPC key describes the complete underlying allocation. Exporting an
  // interior pointer would lose its offset because FlagCX transports only the
  // opaque 64-byte handle.
  if (allocationBase != devPtr || allocationSize == 0) {
    WARN("cannAdaptorIpcMemHandleGet: allocation is not exportable "
         "devPtr=%p allocationBase=%p allocationSize=%zu",
         devPtr, allocationBase, allocationSize);
    return flagcxInvalidArgument;
  }

  CannDevicePtrKey allocationKey{
      device, reinterpret_cast<uintptr_t>(allocationBase)};

  std::lock_guard<std::mutex> lock(cannIpcMutex);
  auto existing = cannExportedIpcKeys.find(allocationKey);
  if (existing != cannExportedIpcKeys.end()) {
    existing->second.refCount++;
    memcpy(handle->key, existing->second.key.data(),
           FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES);
    return flagcxSuccess;
  }

  CannIpcKey key = {};
  ret = aclrtIpcMemGetExportKey(
      allocationBase, allocationSize, key.data(), key.size(),
      ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT);
  if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemHandleGet: aclrtIpcMemGetExportKey failed "
         "ret=%d device=%d allocationBase=%p allocationSize=%zu flags=%d",
         static_cast<int>(ret), device, allocationBase, allocationSize,
         static_cast<int>(ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT));
    return flagcxUnhandledDeviceError;
  }
  key[FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES] = '\0';
  memcpy(handle->key, key.data(), FLAGCX_ASCEND_IPC_KEY_STORAGE_BYTES);
  cannExportedIpcKeys.emplace(
      allocationKey,
      CannExportedIpcRecord{key, static_cast<size_t>(1)});
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == nullptr || devPtr == nullptr)
    return flagcxInvalidArgument;
  *devPtr = nullptr;

  CannIpcKey key = cannRebuildIpcKey(handle);
  if (key[0] == '\0')
    return flagcxInvalidArgument;

  int device = 0;
  FLAGCXCHECK(cannGetCurrentDevice(&device));
  CannImportKey importKey{device, cannSerializedIpcKey(key)};

  std::lock_guard<std::mutex> lock(cannIpcMutex);
  auto existing = cannImportedIpcKeys.find(importKey);
  if (existing != cannImportedIpcKeys.end()) {
    auto ptrKey = CannDevicePtrKey{
        device, reinterpret_cast<uintptr_t>(existing->second)};
    auto record = cannImportedIpcPtrs.find(ptrKey);
    if (record == cannImportedIpcPtrs.end())
      return flagcxInternalError;
    record->second.refCount++;
    *devPtr = existing->second;
    return flagcxSuccess;
  }

  void *importedPtr = nullptr;
  aclError ret = aclrtIpcMemImportByKey(
      &importedPtr, key.data(),
      ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT);
  if (ret != ACL_SUCCESS) {
    WARN("cannAdaptorIpcMemHandleOpen: aclrtIpcMemImportByKey failed "
         "ret=%d device=%d flags=%d",
         static_cast<int>(ret), device,
         static_cast<int>(ACL_RT_IPC_MEM_IMPORT_FLAG_DEFAULT));
    return flagcxUnhandledDeviceError;
  }
  if (importedPtr == nullptr) {
    aclrtIpcMemClose(key.data());
    return flagcxUnhandledDeviceError;
  }

  CannDevicePtrKey ptrKey{device,
                          reinterpret_cast<uintptr_t>(importedPtr)};
  if (cannImportedIpcPtrs.find(ptrKey) != cannImportedIpcPtrs.end()) {
    aclrtIpcMemClose(key.data());
    return flagcxInternalError;
  }
  cannImportedIpcPtrs.emplace(
      ptrKey, CannImportedIpcRecord{key, static_cast<size_t>(1)});
  cannImportedIpcKeys.emplace(importKey, importedPtr);
  *devPtr = importedPtr;
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == nullptr)
    return flagcxInvalidArgument;

  int device = 0;
  FLAGCXCHECK(cannGetCurrentDevice(&device));
  CannDevicePtrKey ptrKey{device, reinterpret_cast<uintptr_t>(devPtr)};

  std::lock_guard<std::mutex> lock(cannIpcMutex);
  auto record = cannImportedIpcPtrs.find(ptrKey);
  if (record == cannImportedIpcPtrs.end()) {
    // ACL requires both importers and the exporter to close the shared key.
    // Generic teardown first closes imported peer pointers, then passes the
    // local allocation base through this same adaptor entry point.
    auto exported = cannExportedIpcKeys.find(ptrKey);
    if (exported == cannExportedIpcKeys.end())
      return flagcxInvalidArgument;
    if (exported->second.refCount > 1) {
      exported->second.refCount--;
      return flagcxSuccess;
    }
    aclError ret = aclrtIpcMemClose(exported->second.key.data());
    if (ret != ACL_SUCCESS)
      return flagcxUnhandledDeviceError;
    cannExportedIpcKeys.erase(exported);
    return flagcxSuccess;
  }
  if (record->second.refCount > 1) {
    record->second.refCount--;
    return flagcxSuccess;
  }

  aclError ret = aclrtIpcMemClose(record->second.key.data());
  if (ret != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  CannImportKey importKey{device,
                          cannSerializedIpcKey(record->second.key)};
  cannImportedIpcKeys.erase(importKey);
  cannImportedIpcPtrs.erase(record);
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != nullptr)
    free(handle);
  // This releases only FlagCX's temporary serialized-key wrapper. The ACL
  // exporter stays alive until the owning allocation is released.
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(
        aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cannAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  (void)flags;
  if (stream == nullptr || addr == nullptr ||
      (reinterpret_cast<uintptr_t>(addr) & (sizeof(uint64_t) - 1)) != 0)
    return flagcxInvalidArgument;
  // The GEQ flag is 0 in both CANN spellings: 8.5 documentation calls it
  // ACL_VALUE_WAIT_GEQ while newer public headers use
  // ACL_STREAM_WAIT_VALUE_GEQ. Pass the ABI value to stay source-compatible
  // across those toolkit header revisions.
  DEVCHECK(aclrtValueWait(addr, value, 0U, stream->base));
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == nullptr || addr == nullptr ||
      (reinterpret_cast<uintptr_t>(addr) & (sizeof(uint64_t) - 1)) != 0)
    return flagcxInvalidArgument;
  DEVCHECK(aclrtValueWrite(addr, value, 0, stream->base));
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                           flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t cannAdaptorHostRegister(void *ptr, size_t size) {
  // Explicit registrations are used for mmap-backed shared memory. Pin the
  // pages in addition to mapping them into the Device address space.
  return cannRegisterHostMemory(
      ptr, size, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED);
}
flagcxResult_t cannAdaptorHostUnregister(void *ptr) {
  return cannUnregisterHostMemory(ptr);
}

// Symmetric memory VMM stubs (not supported)
flagcxResult_t cannAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                       size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t cannAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                     void **) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastCreate(size_t, int, const int *, void **,
                                             int *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                           int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cannAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t cannAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
}

struct flagcxDeviceAdaptor cannAdaptor {
  "CANN",
      // Basic functions
      cannAdaptorDeviceSynchronize, cannAdaptorDeviceMemcpy,
      cannAdaptorDeviceMemset, cannAdaptorDeviceMalloc, cannAdaptorDeviceFree,
      cannAdaptorSetDevice, cannAdaptorGetDevice, cannAdaptorGetDeviceCount,
      cannAdaptorGetVendor, cannAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cannAdaptorGdrMemAlloc, cannAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      cannAdaptorStreamCreate, cannAdaptorStreamDestroy, cannAdaptorStreamCopy,
      cannAdaptorStreamFree, cannAdaptorStreamSynchronize,
      cannAdaptorStreamQuery, cannAdaptorStreamWaitEvent,
      cannAdaptorStreamWaitValue64, cannAdaptorStreamWriteValue64,
      // Event functions
      cannAdaptorEventCreate, cannAdaptorEventDestroy, cannAdaptorEventRecord,
      cannAdaptorEventSynchronize, cannAdaptorEventQuery,
      cannAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cannAdaptorIpcMemHandleCreate, cannAdaptorIpcMemHandleGet,
      cannAdaptorIpcMemHandleOpen, cannAdaptorIpcMemHandleClose,
      cannAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      NULL, // flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps
            // *props, int dev);
      cannAdaptorGetDevicePciBusId,
      cannAdaptorGetDeviceByPciBusId,
      cannAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      cannAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      cannAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      cannAdaptorSymPhysAlloc, cannAdaptorSymPhysFree, cannAdaptorSymFlatMap,
      cannAdaptorSymFlatUnmap, cannAdaptorSymMulticastSupported,
      cannAdaptorSymMulticastCreate, cannAdaptorSymMulticastBind,
      cannAdaptorSymMulticastTeardown, cannAdaptorSymMulticastFree,
      // Explicit ACL IPC security and peer-access setup
      cannAdaptorIpcMemGetProcessId,
      cannAdaptorIpcMemHandleSetImportPid,
      cannAdaptorIpcMemEnablePeerAccess,
};

#endif // USE_ASCEND_ADAPTOR
