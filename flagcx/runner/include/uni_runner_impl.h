#ifndef FLAGCX_UNIRUNNER_IMPL_H_
#define FLAGCX_UNIRUNNER_IMPL_H_

#include "device.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include "utils.h"
#include <memory>
#include <pthread.h>

// DAG node types
typedef enum {
  uniRunnerDagNodeTypeP2p = 0,
  uniRunnerDagNodeTypeRed = 1,
  uniRunnerDagNodeTypeCpy = 2,
  uniRunnerDagNodeTypeIpc = 3
} uniRunnerDagNodeType;

// Static DAG template algorithm identifiers used by the uniRunner cache.
typedef enum {
  uniRunnerDagAlgoDummy = 0,
  uniRunnerDagAlgoLocRed = 1,
  uniRunnerDagAlgoGroupedAG = 2,
  uniRunnerDagAlgoRingAG = 3,
  uniRunnerDagAlgoRingAR = 4,
  uniRunnerDagAlgoSlicedAR = 5,
  uniRunnerDagAlgoRingRS = 6,
  uniRunnerDagAlgoTreeRed = 7,
  uniRunnerDagAlgoIpcAR = 8,
  uniRunnerDagAlgoDirectA2A = 9,
  uniRunnerDagAlgoIpcA2A = 10
} uniRunnerDagAlgoType;

// Cache key describing a reusable uniRunner DAG template.
struct uniRunnerDagCacheKey {
  int formatVersion;
  uniRunnerDagAlgoType algoType;
  flagcxCommOp_t commOp;
  size_t count;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;
  int rank;
  int nranks;
  int root;
};

// Single P2P operation data
struct uniRunnerP2pOpData {
  void *addr;                // Buffer address
  size_t count;              // Element count
  int peerRank;              // Peer rank
  flagcxDataType_t datatype; // Data type
  flagcxDevicePrim type;     // Primitive type (send/recv/term/wait)
};

// P2P node data (supports multiple operations in a group)
struct uniRunnerP2pNodeData {
  struct uniRunnerP2pOpData *ops; // Array of P2P operations
  int numOps;                     // Number of operations
};

// Reduce node data (operation-specific fields only)
struct uniRunnerRedNodeData {
  void *input1;
  void *input2;
  void *output;
  size_t count;
  size_t nthreads;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;

  // Trigger and state tracking
  int triggerIdx; // Trigger index in FIFO
};

// Copy node data (operation-specific fields only)
struct uniRunnerCpyNodeData {
  void *src;
  void *dst;
  size_t count;
  flagcxDataType_t datatype;
};

struct uniRunnerIpcNodeData {
  size_t srcOffsetBytes;
  size_t dstOffsetBytes;
  size_t bytes;
  flagcxIpcBufferType srcBufferType;
  int peerLocalRank;
  uint32_t readySlot;
  uint32_t parentFlagsOffset;
  int triggerIdx;
};

// Unified DAG node with common DAG structure fields
struct uniRunnerDagNode {
  uniRunnerDagNodeType nodeType; // Discriminator for union

  // Common DAG structure fields (shared by all node types)
  int nodeIdx;                   // Unique index of the node in the DAG
  int numParents;                // Number of parent dependencies
  int *parents;                  // Array of parent node indices
  int pendingParents;            // Remaining parents before host submission
  int numChildren;               // Number of children
  int *children;                 // Array of child node indices
  struct uniRunnerDagNode *next; // Queue linkage

  // Union for type-specific operation data
  union {
    struct uniRunnerP2pNodeData p2p;
    struct uniRunnerRedNodeData red;
    struct uniRunnerCpyNodeData cpy;
    struct uniRunnerIpcNodeData ipc;
  } nodeData;
};

typedef struct {
  pthread_t thread;
  flagcxFifo_t fifo;
  flagcxFifo_t ipcFifo;
  void *ipcFifoDevicePtr;
  flagcxStream_t commStream;
  flagcxStream_t redStream;
  flagcxStream_t cpyStream;

  // new: DAG and scheduling queues
  struct uniRunnerDagNode *dagNodes; // Array of all DAG nodes
  int numDagNodes;
  int numPendingNodes;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pReadyQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redReadyQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      ipcReadyQueue;

  uint64_t uniRunnerNSlices;
  uint64_t uniRunnerNThreads;
  uint64_t uniRunnerNRedBlocks;
  uint64_t uniRunnerNIpcBlocks;
  uint64_t uniRunnerIpcChunkSize;
  uint64_t uniRunnerNRedSlices;
  uint64_t uniRunnerRedSliceSize;
  // One launch-wide divisor used by terminal Avg RED nodes. Intermediate
  // Avg nodes are materialized as Sum nodes, so no trigger needs to carry it.
  uint64_t avgDivisor;

  // Stream completion flags backed by a reusable contiguous device pool. The
  // host-side queue stores per-node addresses within that pool.
  void *streamFlagsPool;
  void **streamFlags;
  size_t streamFlagsSize;
  size_t streamFlagsCapacity;

  // IPC/LSA collective runtime resources. User-buffer DevMem views are reused
  // while their base/size bindings remain unchanged; owned ready storage is
  // communicator scoped and reused with monotonically increasing epochs.
  flagcxComm_t ipcOwner;
  flagcxDevMem_t ipcInputMem;
  flagcxDevMem_t ipcOutputMem;
  flagcxDevMem_t ipcReadyMem;
  const void *ipcInputBase;
  void *ipcOutputBase;
  size_t ipcDataBytes;
  void *ipcReadyBuffer;
  size_t ipcReadyCapacity;
  size_t ipcReadySlots;
  uint64_t ipcEpoch;
  uint64_t *ipcParentFlagsDevice;
  size_t ipcParentFlagsCount;
} flagcxUniRunnerState;

flagcxResult_t initUniRunnerStateDummy(flagcxUniRunnerState *runnerState);
flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateGroupedAG(flagcxUniRunnerState *runnerState,
                                           const void *sendbuff, void *recvbuff,
                                           size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxComm_t comm, int groupSize);
flagcxResult_t initUniRunnerStateAlltoAll(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxComm_t comm);
flagcxResult_t initUniRunnerStateIpcA2A(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxComm_t comm);
flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateIpcAR(flagcxUniRunnerState *runnerState,
                                       const void *sendbuff, void *recvbuff,
                                       size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateTreeRed(flagcxUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         void *scratchbuff, size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, int root,
                                         flagcxComm_t comm);
size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key);
flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream);

// Validate public reduction arguments before allocating buffers or publishing
// any RED node to the FIFO.
flagcxResult_t validateUniRunnerReduceArgs(size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxRedOp_t op);
flagcxResult_t checkedUniRunnerTypeBytes(size_t count, size_t multiplier,
                                         flagcxDataType_t datatype,
                                         size_t *bytes);

flagcxResult_t cleanupUniRunner(flagcxComm_t comm);
flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState);
flagcxResult_t runUniRunner(flagcxComm_t comm);
#endif // FLAGCX_UNIRUNNER_IMPL_H_
