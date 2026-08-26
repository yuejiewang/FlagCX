/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "uni_runner_helper.h"

#include "alloc.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <vector>

namespace {

enum uniRunnerDagPlanPhase {
  uniRunnerDagPlanPhaseHost = 0,
  uniRunnerDagPlanPhaseRed = 1,
  uniRunnerDagPlanPhaseIpc = 2,
  uniRunnerDagPlanPhaseRingStep = 3,
  uniRunnerDagPlanNumPhases = 4
};

static bool getUniRunnerDagPlanPhase(uniRunnerDagNodeType nodeType,
                                     int *phase) {
  switch (nodeType) {
    case uniRunnerDagNodeTypeP2p:
    case uniRunnerDagNodeTypeCpy:
      *phase = uniRunnerDagPlanPhaseHost;
      return true;
    case uniRunnerDagNodeTypeRed:
      *phase = uniRunnerDagPlanPhaseRed;
      return true;
    case uniRunnerDagNodeTypeIpc:
      *phase = uniRunnerDagPlanPhaseIpc;
      return true;
    case uniRunnerDagNodeTypeRingStep:
      *phase = uniRunnerDagPlanPhaseRingStep;
      return true;
    default:
      return false;
  }
}

static bool isValidCompiledBufferRef(const uniRunnerDagBufferRef &ref,
                                     size_t accessBytes) {
  if (ref.offsetBytes < 0) {
    return false;
  }
  switch (ref.bufferType) {
    case uniRunnerDagBufferTypeInput:
    case uniRunnerDagBufferTypeOutput:
    case uniRunnerDagBufferTypeScratch:
      return true;
    case uniRunnerDagBufferTypeNone:
      return accessBytes == 0 && ref.offsetBytes == 0;
    default:
      return false;
  }
}

static bool checkedCompiledTypeBytes(size_t count, flagcxDataType_t datatype,
                                     size_t *bytes) {
  const int datatypeValue = static_cast<int>(datatype);
  if (bytes == NULL || datatypeValue < 0 || datatypeValue >= flagcxNumTypes) {
    return false;
  }
  const size_t typeSize = getFlagcxDataTypeSize(datatype);
  if (typeSize == 0 || count > std::numeric_limits<size_t>::max() / typeSize) {
    return false;
  }
  *bytes = count * typeSize;
  return true;
}

static bool isValidP2pPrim(flagcxDevicePrim prim) {
  return prim == flagcxDevicePrimSend || prim == flagcxDevicePrimRecv;
}

static bool isValidCompiledKey(const uniRunnerDagCacheKey &key) {
  const int datatype = static_cast<int>(key.datatype);
  const int redOp = static_cast<int>(key.redOp);
  return key.commOp >= flagcxCommOpSend && key.commOp < flagcxCommNoOp &&
         datatype >= 0 &&
         datatype < flagcxNumTypes && getFlagcxDataTypeSize(key.datatype) != 0 &&
         (redOp == flagcxRedNoOp ||
          (redOp >= static_cast<int>(flagcxSum) &&
           redOp < static_cast<int>(flagcxNumRedOps))) &&
         key.rank >= 0 && key.nranks > 0 && key.rank < key.nranks;
}

static uint64_t makeDagEdge(int parentIdx, int childIdx) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(parentIdx)) << 32) |
         static_cast<uint32_t>(childIdx);
}

template <typename NodeAccessor>
flagcxResult_t compileUniRunnerDagOrder(
    size_t numNodes, const NodeAccessor &nodeAt, std::vector<int> *topoOrder,
    size_t *numHostNodes, size_t *numRedNodes, size_t *numIpcNodes,
    size_t *numRingStepNodes) {
  if (topoOrder == NULL || numHostNodes == NULL || numRedNodes == NULL ||
      numIpcNodes == NULL || numRingStepNodes == NULL ||
      numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  topoOrder->clear();
  *numHostNodes = 0;
  *numRedNodes = 0;
  *numIpcNodes = 0;
  *numRingStepNodes = 0;
  if (numNodes == 0) {
    return flagcxSuccess;
  }

  try {
    std::vector<int> indegree(numNodes, 0);
    std::vector<int> nextReady(numNodes, -1);
    std::array<int, uniRunnerDagPlanNumPhases> readyHead = {{-1, -1, -1, -1}};
    std::array<int, uniRunnerDagPlanNumPhases> readyTail = {{-1, -1, -1, -1}};
    std::unordered_set<uint64_t> parentEdges;
    size_t numEdges = 0;

    const auto enqueueReady = [&](int phase, int nodeIdx) {
      if (readyTail[phase] == -1) {
        readyHead[phase] = nodeIdx;
      } else {
        nextReady[readyTail[phase]] = nodeIdx;
      }
      readyTail[phase] = nodeIdx;
    };

    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      const auto &node = nodeAt(nodeIdx);
      int phase = -1;
      if (node.nodeIdx != static_cast<int>(nodeIdx) ||
          !getUniRunnerDagPlanPhase(node.nodeType, &phase)) {
        return flagcxInternalError;
      }
      const size_t numParents = node.numParents();
      const size_t numChildren = node.numChildren();
      if (numParents > static_cast<size_t>(std::numeric_limits<int>::max()) ||
          numChildren > static_cast<size_t>(std::numeric_limits<int>::max()) ||
          numParents > std::numeric_limits<size_t>::max() - numEdges) {
        return flagcxInvalidArgument;
      }
      numEdges += numParents;
      indegree[nodeIdx] = static_cast<int>(numParents);
      if (numParents == 0) {
        enqueueReady(phase, static_cast<int>(nodeIdx));
      }
    }

    parentEdges.reserve(numEdges);
    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      const auto &node = nodeAt(nodeIdx);
      for (size_t parentSlot = 0; parentSlot < node.numParents();
           ++parentSlot) {
        const int parentIdx = node.parent(parentSlot);
        if (parentIdx < 0 || static_cast<size_t>(parentIdx) >= numNodes ||
            parentIdx == static_cast<int>(nodeIdx) ||
            !parentEdges.emplace(
                makeDagEdge(parentIdx, static_cast<int>(nodeIdx))).second) {
          return flagcxInternalError;
        }
      }
    }

    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      const auto &node = nodeAt(nodeIdx);
      for (size_t childSlot = 0; childSlot < node.numChildren();
           ++childSlot) {
        const int childIdx = node.child(childSlot);
        if (childIdx < 0 || static_cast<size_t>(childIdx) >= numNodes ||
            childIdx == static_cast<int>(nodeIdx)) {
          return flagcxInternalError;
        }
        const std::unordered_set<uint64_t>::iterator edge =
            parentEdges.find(makeDagEdge(static_cast<int>(nodeIdx), childIdx));
        if (edge == parentEdges.end()) {
          return flagcxInternalError;
        }
        parentEdges.erase(edge);
      }
    }
    if (!parentEdges.empty()) {
      return flagcxInternalError;
    }

    topoOrder->reserve(numNodes);
    while (topoOrder->size() < numNodes) {
      const size_t orderedBeforeRound = topoOrder->size();
      for (int phase = 0; phase < uniRunnerDagPlanNumPhases; ++phase) {
        while (readyHead[phase] != -1) {
          const int nodeIdx = readyHead[phase];
          readyHead[phase] = nextReady[nodeIdx];
          nextReady[nodeIdx] = -1;
          if (readyHead[phase] == -1) {
            readyTail[phase] = -1;
          }

          if (topoOrder->size() >= numNodes) {
            return flagcxInternalError;
          }
          topoOrder->push_back(nodeIdx);
          if (phase == uniRunnerDagPlanPhaseHost) {
            ++*numHostNodes;
          } else if (phase == uniRunnerDagPlanPhaseRed) {
            ++*numRedNodes;
          } else if (phase == uniRunnerDagPlanPhaseIpc) {
            ++*numIpcNodes;
          } else {
            ++*numRingStepNodes;
          }

          const auto &node = nodeAt(static_cast<size_t>(nodeIdx));
          for (size_t childSlot = 0; childSlot < node.numChildren();
               ++childSlot) {
            const int childIdx = node.child(childSlot);
            if (indegree[childIdx] <= 0) {
              return flagcxInternalError;
            }
            --indegree[childIdx];
            if (indegree[childIdx] == 0) {
              int childPhase = -1;
              if (!getUniRunnerDagPlanPhase(
                      nodeAt(static_cast<size_t>(childIdx)).nodeType,
                      &childPhase)) {
                return flagcxInternalError;
              }
              enqueueReady(childPhase, childIdx);
            }
          }
        }
      }
      if (topoOrder->size() == orderedBeforeRound) {
        return flagcxInvalidArgument;
      }
    }

    return *numHostNodes + *numRedNodes + *numIpcNodes +
                   *numRingStepNodes ==
               numNodes
               ? flagcxSuccess
               : flagcxInternalError;
  } catch (...) {
    topoOrder->clear();
    *numHostNodes = 0;
    *numRedNodes = 0;
    *numIpcNodes = 0;
    *numRingStepNodes = 0;
    return flagcxSystemError;
  }
}

struct RuntimeNodeAccessor {
  explicit RuntimeNodeAccessor(const uniRunnerDagNode *nodes) : nodes(nodes) {}
  struct View {
    explicit View(const uniRunnerDagNode &node)
        : nodeIdx(node.nodeIdx), nodeType(node.nodeType), node(node) {}
    size_t numParents() const { return static_cast<size_t>(node.numParents); }
    size_t numChildren() const { return static_cast<size_t>(node.numChildren); }
    int parent(size_t slot) const { return node.parents[slot]; }
    int child(size_t slot) const { return node.children[slot]; }
    int nodeIdx;
    uniRunnerDagNodeType nodeType;
    const uniRunnerDagNode &node;
  };
  View operator()(size_t nodeIdx) const { return View(nodes[nodeIdx]); }
  const uniRunnerDagNode *nodes;
};

struct TemplateNodeAccessor {
  explicit TemplateNodeAccessor(const std::vector<uniRunnerDagNodeDesc> &nodes)
      : nodes(nodes) {}
  struct View {
    explicit View(const uniRunnerDagNodeDesc &node)
        : nodeIdx(node.nodeIdx), nodeType(node.nodeType), node(node) {}
    size_t numParents() const { return node.parents.size(); }
    size_t numChildren() const { return node.children.size(); }
    int parent(size_t slot) const { return node.parents[slot]; }
    int child(size_t slot) const { return node.children[slot]; }
    int nodeIdx;
    uniRunnerDagNodeType nodeType;
    const uniRunnerDagNodeDesc &node;
  };
  View operator()(size_t nodeIdx) const { return View(nodes[nodeIdx]); }
  const std::vector<uniRunnerDagNodeDesc> &nodes;
};

static const uniRunnerRingStepNodeData &
getRingStepPayload(const RuntimeNodeAccessor::View &view) {
  return view.node.nodeData.ringStep;
}

static const uniRunnerDagRingStepDesc &
getRingStepPayload(const TemplateNodeAccessor::View &view) {
  return view.node.ringStep;
}

template <typename NodeAccessor>
flagcxResult_t validateUniRunnerRingLanes(
    size_t numNodes, const NodeAccessor &nodeAt, size_t numHostNodes,
    size_t numRedNodes, size_t numIpcNodes, size_t numRingStepNodes,
    std::vector<uint32_t> *laneOffsets) {
  laneOffsets->clear();
  if (numRingStepNodes == 0) {
    return flagcxSuccess;
  }
  if (numRingStepNodes != numNodes || numHostNodes != 0 || numRedNodes != 0 ||
      numIpcNodes != 0 || numNodes > std::numeric_limits<uint32_t>::max()) {
    return flagcxInvalidArgument;
  }

  try {
    laneOffsets->push_back(0);
    uint32_t expectedChannel = 0;
    uint32_t expectedOrdinal = 0;
    size_t laneStart = 0;
    for (size_t i = 0; i < numNodes; ++i) {
      const auto node = nodeAt(i);
      const auto &step = getRingStepPayload(node);
      if (node.nodeType != uniRunnerDagNodeTypeRingStep ||
          step.channelId != expectedChannel ||
          step.laneOrdinal != expectedOrdinal || step.kind > uniRunnerRingStepRecv ||
          step.offsetElements >
              std::numeric_limits<uint64_t>::max() - step.countElements) {
        return flagcxInvalidArgument;
      }
      const bool first = expectedOrdinal == 0;
      const bool last = i + 1 == numNodes ||
                        getRingStepPayload(nodeAt(i + 1)).channelId !=
                            expectedChannel;
      if ((first ? node.numParents() != 0
                 : node.numParents() != 1 || node.parent(0) != int(i - 1)) ||
          (last ? node.numChildren() != 0
                : node.numChildren() != 1 || node.child(0) != int(i + 1))) {
        return flagcxInvalidArgument;
      }
      if (!first && getRingStepPayload(nodeAt(i - 1)).channelId !=
                        expectedChannel) {
        return flagcxInvalidArgument;
      }
      if (last) {
        const size_t laneNodes = i + 1 - laneStart;
        size_t period = 0;
        for (size_t p = 0; p < laneNodes; ++p) {
          if (getRingStepPayload(nodeAt(laneStart + p)).kind ==
              uniRunnerRingStepRecv) {
            period = p + 1;
            break;
          }
        }
        if (period < 3 || (period & 1) == 0 || laneNodes % period != 0) {
          return flagcxInvalidArgument;
        }
        const size_t nranks = (period + 1) / 2;
        for (size_t p = 0; p < laneNodes; ++p) {
          const size_t q = p % period;
          uint32_t expectedKind = uniRunnerRingStepRecv;
          if (q == 0) {
            expectedKind = uniRunnerRingStepSend;
          } else if (q < nranks - 1) {
            expectedKind = uniRunnerRingStepRecvReduceSend;
          } else if (q == nranks - 1) {
            expectedKind = uniRunnerRingStepRecvReduceCopySend;
          } else if (q < period - 1) {
            expectedKind = uniRunnerRingStepRecvCopySend;
          }
          const auto &payload = getRingStepPayload(nodeAt(laneStart + p));
          if (payload.kind != expectedKind ||
              payload.postOp !=
                  (expectedKind == uniRunnerRingStepRecvReduceCopySend ? 1u
                                                                       : 0u)) {
            return flagcxInvalidArgument;
          }
        }
        laneOffsets->push_back(static_cast<uint32_t>(i + 1));
        ++expectedChannel;
        expectedOrdinal = 0;
        laneStart = i + 1;
      } else {
        ++expectedOrdinal;
      }
    }
    return expectedChannel == 0 ? flagcxInvalidArgument : flagcxSuccess;
  } catch (...) {
    laneOffsets->clear();
    return flagcxSystemError;
  }
}

static flagcxResult_t
validateUniRunnerDagTemplatePayload(const uniRunnerDagTemplate &dagTemplate) {
  if (!isValidCompiledKey(dagTemplate.key) ||
      dagTemplate.nodes.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      (dagTemplate.hashValue != 0 &&
       dagTemplate.hashValue !=
           getUniRunnerDagPatternHash(dagTemplate.key))) {
    return flagcxInvalidArgument;
  }
  size_t numIpcNodes = 0;
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    if (node.nodeType == uniRunnerDagNodeTypeIpc) {
      ++numIpcNodes;
    }
  }
  std::vector<uint8_t> ipcReadySlots;
  try {
    ipcReadySlots.assign(numIpcNodes, uint8_t{0});
  } catch (...) {
    return flagcxSystemError;
  }
  for (const uniRunnerDagNodeDesc &node : dagTemplate.nodes) {
    switch (node.nodeType) {
      case uniRunnerDagNodeTypeP2p:
        for (const uniRunnerDagP2pOpDesc &op : node.p2pOps) {
          size_t bytes = 0;
          if (!checkedCompiledTypeBytes(op.count, op.datatype, &bytes) ||
              !isValidP2pPrim(op.type) || op.peerRank < 0 ||
              op.peerRank >= dagTemplate.key.nranks ||
              !isValidCompiledBufferRef(op.buffer, bytes)) {
            return flagcxInvalidArgument;
          }
        }
        break;
      case uniRunnerDagNodeTypeRed: {
        size_t bytes = 0;
        const int redOp = static_cast<int>(node.red.redOp);
        if (!checkedCompiledTypeBytes(node.red.count, node.red.datatype,
                                      &bytes) ||
            redOp < static_cast<int>(flagcxSum) ||
            redOp >= static_cast<int>(flagcxNumRedOps) ||
            node.red.count >
                flagcxTriggerMask(flagcxReduceTriggerBitsCount) ||
            !isValidCompiledBufferRef(node.red.input1, bytes) ||
            !isValidCompiledBufferRef(node.red.input2, bytes) ||
            !isValidCompiledBufferRef(node.red.output, bytes)) {
          return flagcxInvalidArgument;
        }
        break;
      }
      case uniRunnerDagNodeTypeCpy: {
        size_t bytes = 0;
        if (!checkedCompiledTypeBytes(node.cpy.count, node.cpy.datatype,
                                      &bytes) ||
            !isValidCompiledBufferRef(node.cpy.src, bytes) ||
            !isValidCompiledBufferRef(node.cpy.dst, bytes)) {
          return flagcxInvalidArgument;
        }
        break;
      }
      case uniRunnerDagNodeTypeIpc:
        if (node.ipc.srcBufferType != flagcxIpcBufferInput &&
            node.ipc.srcBufferType != flagcxIpcBufferOutput) {
          return flagcxInvalidArgument;
        }
        if (node.ipc.peerLocalRank < 0 ||
            node.ipc.peerLocalRank >= dagTemplate.key.nranks ||
            static_cast<size_t>(node.ipc.readySlot) >= numIpcNodes ||
            ipcReadySlots[node.ipc.readySlot] != 0) {
          return flagcxInvalidArgument;
        }
        ipcReadySlots[node.ipc.readySlot] = 1;
        break;
      case uniRunnerDagNodeTypeRingStep:
        if (node.ringStep.kind > uniRunnerRingStepRecv ||
            node.ringStep.postOp > 1 ||
            node.ringStep.offsetElements >
                std::numeric_limits<uint64_t>::max() -
                    node.ringStep.countElements) {
          return flagcxInvalidArgument;
        }
        break;
      default:
        return flagcxInternalError;
    }
  }
  // Unique slots in [0, numIpcNodes) imply an exact dense assignment.
  return flagcxSuccess;
}

static bool isValidStaticRedTriggerPayload(const uniRunnerRedNodeData &red) {
  return static_cast<int>(red.datatype) >= 0 &&
         static_cast<int>(red.datatype) < flagcxNumTypes &&
         getFlagcxDataTypeSize(red.datatype) != 0 &&
         static_cast<int>(red.redOp) >= static_cast<int>(flagcxSum) &&
         static_cast<int>(red.redOp) < static_cast<int>(flagcxNumRedOps) &&
         red.count <= flagcxTriggerMask(flagcxReduceTriggerBitsCount) &&
         red.nthreads != 0 &&
         red.nthreads <= flagcxTriggerMask(flagcxReduceTriggerBitsNThreads);
}

static bool isValidStaticIpcBufferType(flagcxIpcBufferType bufferType) {
  return bufferType == flagcxIpcBufferInput ||
         bufferType == flagcxIpcBufferOutput;
}

} // namespace

flagcxResult_t normalizeUniRunnerIpcChunkSize(int64_t configuredChunkSize,
                                              size_t *chunkSize) {
  if (chunkSize == NULL) {
    return flagcxInvalidArgument;
  }
  *chunkSize = 0;
  if (configuredChunkSize < 16 || configuredChunkSize % 16 != 0 ||
      static_cast<uint64_t>(configuredChunkSize) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return flagcxInvalidArgument;
  }
  *chunkSize = static_cast<size_t>(configuredChunkSize);
  return flagcxSuccess;
}

flagcxResult_t checkedUniRunnerIpcChunkCount(size_t bytes, size_t chunkSize,
                                             uint32_t *numChunks) {
  if (numChunks == NULL) {
    return flagcxInvalidArgument;
  }
  *numChunks = 0;
  if (chunkSize < 16 || chunkSize % 16 != 0) {
    return flagcxInvalidArgument;
  }
  size_t chunks = bytes / chunkSize + (bytes % chunkSize != 0);
  if (chunks == 0) {
    chunks = 1;
  }
  if (chunks > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return flagcxInvalidArgument;
  }
  *numChunks = static_cast<uint32_t>(chunks);
  return flagcxSuccess;
}

void destroyUniRunnerDagExecutionPlan(uniRunnerDagExecutionPlan *plan) {
  if (plan == NULL) {
    return;
  }
  if (plan->ownsTopoOrder && plan->topoOrder != NULL) {
    free(const_cast<int *>(plan->topoOrder));
  }
  if (plan->ownsRingLaneOffsets && plan->ringLaneOffsets != NULL) {
    free(const_cast<uint32_t *>(plan->ringLaneOffsets));
  }
  *plan = {};
}

flagcxResult_t compileUniRunnerDagExecutionPlan(
    const uniRunnerDagNode *nodes, size_t numNodes,
    uniRunnerDagExecutionPlan *plan) {
  if (plan == NULL) {
    return flagcxInvalidArgument;
  }
  destroyUniRunnerDagExecutionPlan(plan);

  if (numNodes >
      static_cast<size_t>(std::numeric_limits<int>::max())) {
    return flagcxInvalidArgument;
  }
  if (numNodes == 0) {
    return flagcxSuccess;
  }
  if (nodes == NULL) {
    return flagcxInvalidArgument;
  }

  for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    if (nodes[nodeIdx].numParents < 0 || nodes[nodeIdx].numChildren < 0 ||
        (nodes[nodeIdx].numParents > 0 && nodes[nodeIdx].parents == NULL) ||
        (nodes[nodeIdx].numChildren > 0 && nodes[nodeIdx].children == NULL)) {
      return flagcxInternalError;
    }
  }

  std::vector<int> order;
  size_t numHostNodes = 0;
  size_t numRedNodes = 0;
  size_t numIpcNodes = 0;
  size_t numRingStepNodes = 0;
  FLAGCXCHECK(compileUniRunnerDagOrder(
      numNodes, RuntimeNodeAccessor(nodes), &order, &numHostNodes,
      &numRedNodes, &numIpcNodes, &numRingStepNodes));

  std::vector<uint32_t> laneOffsets;
  FLAGCXCHECK(validateUniRunnerRingLanes(
      numNodes, RuntimeNodeAccessor(nodes), numHostNodes, numRedNodes,
      numIpcNodes, numRingStepNodes, &laneOffsets));

  int *topoOrder = NULL;
  FLAGCXCHECK(flagcxCalloc(&topoOrder, numNodes));
  std::copy(order.begin(), order.end(), topoOrder);
  plan->topoOrder = topoOrder;
  plan->numNodes = numNodes;
  plan->numHostNodes = numHostNodes;
  plan->numRedNodes = numRedNodes;
  plan->numIpcNodes = numIpcNodes;
  plan->numRingStepNodes = numRingStepNodes;
  plan->ownsTopoOrder = true;
  if (!laneOffsets.empty()) {
    uint32_t *offsets = NULL;
    flagcxResult_t allocResult = flagcxCalloc(&offsets, laneOffsets.size());
    if (allocResult != flagcxSuccess) {
      destroyUniRunnerDagExecutionPlan(plan);
      return allocResult;
    }
    std::copy(laneOffsets.begin(), laneOffsets.end(), offsets);
    plan->ringLaneOffsets = offsets;
    plan->numRingLanes = laneOffsets.size() - 1;
    plan->ownsRingLaneOffsets = true;
  }
  return flagcxSuccess;
}

flagcxResult_t compileUniRunnerDagTemplate(
    const uniRunnerDagTemplate &dagTemplate,
    uniRunnerCompiledDagTemplate *compiledTemplate) {
  if (compiledTemplate == NULL) {
    return flagcxInvalidArgument;
  }
  try {
    *compiledTemplate = {};
    const size_t numNodes = dagTemplate.nodes.size();
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      return flagcxInvalidArgument;
    }
    FLAGCXCHECK(validateUniRunnerDagTemplatePayload(dagTemplate));
    uniRunnerCompiledDagTemplate compiled;
    FLAGCXCHECK(compileUniRunnerDagOrder(
        numNodes, TemplateNodeAccessor(dagTemplate.nodes),
        &compiled.topoOrder, &compiled.numHostNodes, &compiled.numRedNodes,
        &compiled.numIpcNodes, &compiled.numRingStepNodes));
    FLAGCXCHECK(validateUniRunnerRingLanes(
        numNodes, TemplateNodeAccessor(dagTemplate.nodes),
        compiled.numHostNodes, compiled.numRedNodes, compiled.numIpcNodes,
        compiled.numRingStepNodes, &compiled.ringLaneOffsets));
    compiled.numRingLanes = compiled.ringLaneOffsets.empty()
                                ? 0
                                : compiled.ringLaneOffsets.size() - 1;
    compiled.dagTemplate = dagTemplate;
    compiled.dagTemplate.hashValue =
        getUniRunnerDagPatternHash(compiled.dagTemplate.key);
    compiled.numNodes = numNodes;
    *compiledTemplate = std::move(compiled);
    return flagcxSuccess;
  } catch (...) {
    *compiledTemplate = {};
    return flagcxSystemError;
  }
}

flagcxResult_t resolveUniRunnerStaticExecutorSchedule(
    const uniRunnerDagExecutionPlan *plan, size_t requestedRedBlocks,
    size_t requestedIpcBlocks, size_t maxIpcParallelism,
    size_t maxExecutorBlocks,
    uniRunnerStaticExecutorSchedule *schedule) {
  if (schedule == NULL) {
    return flagcxInvalidArgument;
  }
  *schedule = {};
  if (plan == NULL ||
      plan->numNodes >
          static_cast<size_t>(std::numeric_limits<int>::max()) ||
      ((plan->numNodes == 0) != (plan->topoOrder == NULL)) ||
      plan->numHostNodes > plan->numNodes ||
      plan->numRedNodes > plan->numNodes - plan->numHostNodes ||
      plan->numIpcNodes >
          plan->numNodes - plan->numHostNodes - plan->numRedNodes ||
      plan->numRingStepNodes != plan->numNodes - plan->numHostNodes -
                                        plan->numRedNodes - plan->numIpcNodes ||
      plan->numRingStepNodes != 0) {
    return flagcxInvalidArgument;
  }

  schedule->numRedTasks = plan->numRedNodes;
  schedule->numIpcTasks = plan->numIpcNodes;
  if ((plan->numIpcNodes == 0 && maxIpcParallelism != 0) ||
      (plan->numIpcNodes != 0 && maxIpcParallelism == 0)) {
    *schedule = {};
    return flagcxInvalidArgument;
  }
  if (plan->numRedNodes == 0 && plan->numIpcNodes == 0) {
    return flagcxSuccess;
  }
  if ((plan->numRedNodes != 0 && requestedRedBlocks == 0) ||
      (plan->numIpcNodes != 0 && requestedIpcBlocks == 0)) {
    *schedule = {};
    return flagcxInvalidArgument;
  }

  const size_t desiredRedBlocks =
      std::min(requestedRedBlocks, plan->numRedNodes);
  const size_t desiredIpcBlocks =
      std::min(requestedIpcBlocks, maxIpcParallelism);
  const size_t activeExecutorTypes =
      static_cast<size_t>(desiredRedBlocks != 0) +
      static_cast<size_t>(desiredIpcBlocks != 0);
  if (maxExecutorBlocks < activeExecutorTypes) {
    *schedule = {};
    return flagcxInvalidArgument;
  }

  const size_t desiredBlocks = desiredRedBlocks + desiredIpcBlocks;
  if (desiredBlocks <= maxExecutorBlocks) {
    schedule->numRedBlocks = desiredRedBlocks;
    schedule->numIpcBlocks = desiredIpcBlocks;
    return flagcxSuccess;
  }

  // Preserve at least one block for every non-empty executor type. Split the
  // remaining residency budget approximately evenly, then give unused share
  // to the type that still has requested work (RED first for deterministic
  // tie-breaking). This path is reached only for an over-subscribed config.
  schedule->numRedBlocks = desiredRedBlocks != 0 ? 1 : 0;
  schedule->numIpcBlocks = desiredIpcBlocks != 0 ? 1 : 0;
  const size_t redExtraNeeded =
      desiredRedBlocks - schedule->numRedBlocks;
  const size_t ipcExtraNeeded =
      desiredIpcBlocks - schedule->numIpcBlocks;
  const size_t totalExtraNeeded = redExtraNeeded + ipcExtraNeeded;
  size_t remaining = std::min(
      maxExecutorBlocks - activeExecutorTypes, totalExtraNeeded);

  if (redExtraNeeded != 0 && ipcExtraNeeded != 0) {
    const size_t redShare = remaining / 2 + remaining % 2;
    const size_t redExtra = std::min(redExtraNeeded, redShare);
    schedule->numRedBlocks += redExtra;
    remaining -= redExtra;
    const size_t ipcExtra = std::min(ipcExtraNeeded, remaining);
    schedule->numIpcBlocks += ipcExtra;
    remaining -= ipcExtra;
  }

  const size_t redRemaining =
      desiredRedBlocks - schedule->numRedBlocks;
  const size_t redExtra = std::min(redRemaining, remaining);
  schedule->numRedBlocks += redExtra;
  remaining -= redExtra;
  const size_t ipcRemaining =
      desiredIpcBlocks - schedule->numIpcBlocks;
  schedule->numIpcBlocks += std::min(ipcRemaining, remaining);

  const size_t resolvedBlocks =
      schedule->numRedBlocks + schedule->numIpcBlocks;
  if (resolvedBlocks != std::min(desiredBlocks, maxExecutorBlocks) ||
      (plan->numRedNodes != 0 && schedule->numRedBlocks == 0) ||
      (plan->numIpcNodes != 0 && schedule->numIpcBlocks == 0)) {
    *schedule = {};
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

flagcxResult_t resolveUniRunnerStaticExecutorResidencyBudget(
    bool cooperativeLaunch, bool concurrentKernels, size_t smCount,
    size_t activeBlocksPerSm, size_t maxThreadsPerBlock, size_t nthreads,
    size_t *maxExecutorBlocks) {
  if (maxExecutorBlocks == NULL) {
    return flagcxInvalidArgument;
  }
  *maxExecutorBlocks = 0;
  if (nthreads == 0 || maxThreadsPerBlock == 0 ||
      nthreads > maxThreadsPerBlock) {
    return flagcxInvalidArgument;
  }
  if (!cooperativeLaunch || !concurrentKernels || smCount <= 1 ||
      activeBlocksPerSm == 0) {
    return flagcxNotSupported;
  }
  if (activeBlocksPerSm >
      std::numeric_limits<size_t>::max() / smCount) {
    return flagcxInvalidArgument;
  }

  const size_t residentCapacity = activeBlocksPerSm * smCount;
  const size_t launchRange =
      static_cast<size_t>(std::numeric_limits<int>::max());
  *maxExecutorBlocks =
      std::min(std::min(residentCapacity, smCount - 1), launchRange);
  if (*maxExecutorBlocks == 0) {
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t getUniRunnerStaticTaskAssignment(
    size_t taskOrdinal, size_t numTasks, size_t numBlocks, size_t *blockIdx,
    size_t *blockTaskOrdinal) {
  if (blockIdx == NULL || blockTaskOrdinal == NULL || numBlocks == 0 ||
      taskOrdinal >= numTasks) {
    return flagcxInvalidArgument;
  }
  *blockIdx = taskOrdinal % numBlocks;
  *blockTaskOrdinal = taskOrdinal / numBlocks;
  return flagcxSuccess;
}

flagcxResult_t populateUniRunnerStaticRedTriggers(
    uniRunnerDagNode *nodes, size_t numNodes,
    const uniRunnerDagExecutionPlan *plan, void *const *nodeFlags,
    size_t numNodeFlags, flagcxReduceTrigger *triggers,
    size_t triggerCapacity, size_t *numTriggers) {
  if (numTriggers == NULL) {
    return flagcxInvalidArgument;
  }
  *numTriggers = 0;
  if (plan == NULL || plan->numNodes != numNodes ||
      numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      ((numNodes == 0) != (plan->topoOrder == NULL)) ||
      plan->numHostNodes > numNodes ||
      plan->numRedNodes > numNodes - plan->numHostNodes ||
      plan->numIpcNodes > numNodes - plan->numHostNodes - plan->numRedNodes ||
      plan->numRingStepNodes != numNodes - plan->numHostNodes -
                                        plan->numRedNodes - plan->numIpcNodes) {
    return flagcxInvalidArgument;
  }
  if (plan->numRedNodes == 0) {
    return flagcxSuccess;
  }
  if (nodes == NULL || nodeFlags == NULL || numNodeFlags < numNodes ||
      triggers == NULL || triggerCapacity < plan->numRedNodes) {
    return flagcxInvalidArgument;
  }

  // Validate every dependency address, payload field, and single-parent RED
  // constraint before writing any trigger or publishing triggerIdx.
  size_t redOrdinal = 0;
  for (size_t topoSlot = 0; topoSlot < numNodes; ++topoSlot) {
    const int nodeIdx = plan->topoOrder[topoSlot];
    if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= numNodes ||
        nodes[nodeIdx].nodeIdx != nodeIdx) {
      return flagcxInternalError;
    }
    const uniRunnerDagNode *node = &nodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeRed) {
      continue;
    }
    if (redOrdinal >= plan->numRedNodes || node->numParents < 0 ||
        node->numParents > 1 || nodeFlags[nodeIdx] == NULL ||
        !isValidStaticRedTriggerPayload(node->nodeData.red)) {
      return flagcxInvalidArgument;
    }
    if (node->numParents == 1) {
      if (node->parents == NULL || node->parents[0] < 0 ||
          static_cast<size_t>(node->parents[0]) >= numNodes ||
          nodeFlags[node->parents[0]] == NULL) {
        return flagcxInvalidArgument;
      }
    }
    ++redOrdinal;
  }
  if (redOrdinal != plan->numRedNodes) {
    return flagcxInternalError;
  }

  redOrdinal = 0;
  for (size_t topoSlot = 0; topoSlot < numNodes; ++topoSlot) {
    const int nodeIdx = plan->topoOrder[topoSlot];
    uniRunnerDagNode *node = &nodes[nodeIdx];
    if (node->nodeType != uniRunnerDagNodeTypeRed) {
      continue;
    }
    const uint64_t flagIn =
        node->numParents == 0
            ? 0
            : reinterpret_cast<uint64_t>(nodeFlags[node->parents[0]]);
    const uint64_t flagOut =
        reinterpret_cast<uint64_t>(nodeFlags[nodeIdx]);
    triggers[redOrdinal].setValue(
        reinterpret_cast<uint64_t>(node->nodeData.red.input1),
        reinterpret_cast<uint64_t>(node->nodeData.red.input2),
        reinterpret_cast<uint64_t>(node->nodeData.red.output),
        node->nodeData.red.count, node->nodeData.red.nthreads,
        node->nodeData.red.datatype, node->nodeData.red.redOp,
        flagcxReduceTriggerEnqueued, flagIn, flagOut);
    node->nodeData.red.triggerIdx = static_cast<int>(redOrdinal++);
  }
  __sync_synchronize();
  *numTriggers = redOrdinal;
  return flagcxSuccess;
}

flagcxResult_t populateUniRunnerStaticRingDag(
    uniRunnerDagNode *nodes, size_t numNodes,
    const uniRunnerDagExecutionPlan *plan, void *const *nodeFlags,
    size_t numNodeFlags, flagcxHeteroComm_t hcomm,
    size_t simpleBufferBytes, flagcxRingStepTrigger *triggers,
    size_t triggerCapacity, flagcxRingLaneDesc *lanes, size_t laneCapacity) {
  if (plan == NULL || nodes == NULL || nodeFlags == NULL || hcomm == NULL ||
      triggers == NULL || lanes == NULL || plan->numNodes != numNodes ||
      plan->numRingStepNodes != numNodes || plan->numHostNodes != 0 ||
      plan->numRedNodes != 0 || plan->numIpcNodes != 0 ||
      plan->numRingLanes == 0 || plan->ringLaneOffsets == NULL ||
      triggerCapacity != numNodes || laneCapacity != plan->numRingLanes ||
      numNodeFlags < numNodes || simpleBufferBytes == 0 ||
      hcomm->localRanks != hcomm->nRanks) {
    return flagcxInvalidArgument;
  }
  for (size_t laneId = 0; laneId < plan->numRingLanes; ++laneId) {
    const uint32_t first = plan->ringLaneOffsets[laneId];
    const uint32_t end = plan->ringLaneOffsets[laneId + 1];
    if (first >= end || end > numNodes || laneId >= MAXCHANNELS) {
      return flagcxInvalidArgument;
    }
    const flagcxRing &ring = hcomm->channels[laneId].ring;
    if (ring.index < 0 || ring.index >= hcomm->nRanks || ring.prev < 0 ||
        ring.prev >= hcomm->nRanks || ring.next < 0 ||
        ring.next >= hcomm->nRanks || hcomm->rankToLocalRank == NULL) {
      return flagcxInvalidArgument;
    }
    const int prevLocalRank = hcomm->rankToLocalRank[ring.prev];
    const int nextLocalRank = hcomm->rankToLocalRank[ring.next];
    if (prevLocalRank < 0 || prevLocalRank >= hcomm->localRanks ||
        nextLocalRank < 0 || nextLocalRank >= hcomm->localRanks ||
        laneId > std::numeric_limits<uint64_t>::max() / simpleBufferBytes) {
      return flagcxInvalidArgument;
    }
    flagcxRingLaneDesc &lane = lanes[laneId];
    lane = {};
    lane.firstTrigger = first;
    lane.numTriggers = end - first;
    lane.scratchChannelOffsetBytes = laneId * simpleBufferBytes;
    lane.channelId = static_cast<uint32_t>(laneId);
    lane.ringIndex = static_cast<uint32_t>(ring.index);
    lane.prevLocalRank = prevLocalRank;
    lane.nextLocalRank = nextLocalRank;
    lane.nRanks = static_cast<uint32_t>(hcomm->nRanks);

    for (uint32_t ordinal = first; ordinal < end; ++ordinal) {
      uniRunnerDagNode &node = nodes[ordinal];
      const uniRunnerRingStepNodeData &step = node.nodeData.ringStep;
      if (node.nodeType != uniRunnerDagNodeTypeRingStep ||
          step.channelId != laneId || step.laneOrdinal != ordinal - first) {
        return flagcxInvalidArgument;
      }
      flagcxRingStepTrigger &trigger = triggers[ordinal];
      trigger = {};
      trigger.offsetElements = step.offsetElements;
      trigger.countElements = step.countElements;
      trigger.flagOut = ordinal + 1 == end
                            ? reinterpret_cast<uint64_t>(nodeFlags[ordinal])
                            : 0;
      trigger.channelId = step.channelId;
      trigger.laneOrdinal = step.laneOrdinal;
      trigger.kind = step.kind;
      trigger.postOp = step.postOp;
      node.nodeData.ringStep.triggerIdx = static_cast<int>(ordinal);
    }
  }
  return flagcxSuccess;
}

flagcxResult_t populateUniRunnerStaticIpcTriggers(
    uniRunnerDagNode *nodes, size_t numNodes,
    const uniRunnerDagExecutionPlan *plan, void *const *nodeFlags,
    size_t numNodeFlags, const uniRunnerStaticIpcTriggerConfig *config,
    flagcxIpcTrigger *triggers, size_t triggerCapacity, size_t *numTriggers,
    size_t *maxChunksPerTrigger) {
  if (numTriggers == NULL || maxChunksPerTrigger == NULL) {
    return flagcxInvalidArgument;
  }
  *numTriggers = 0;
  *maxChunksPerTrigger = 0;
  if (plan == NULL || plan->numNodes != numNodes ||
      numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      ((numNodes == 0) != (plan->topoOrder == NULL)) ||
      plan->numHostNodes > numNodes ||
      plan->numRedNodes > numNodes - plan->numHostNodes ||
      plan->numIpcNodes > numNodes - plan->numHostNodes - plan->numRedNodes ||
      plan->numRingStepNodes != numNodes - plan->numHostNodes -
                                        plan->numRedNodes - plan->numIpcNodes) {
    return flagcxInvalidArgument;
  }
  if (plan->numIpcNodes == 0) {
    return flagcxSuccess;
  }
  if (nodes == NULL || nodeFlags == NULL || numNodeFlags < numNodes ||
      config == NULL || triggers == NULL ||
      triggerCapacity != plan->numIpcNodes || config->chunkSize < 16 ||
      config->chunkSize % 16 != 0 || config->epoch == 0 ||
      config->readySlots != plan->numIpcNodes || config->localRanks <= 0 ||
      config->parentFlagsCount >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    return flagcxInvalidArgument;
  }

  try {
    std::vector<unsigned char> seenNodes(numNodes, 0);
    std::vector<unsigned char> seenReadySlots(config->readySlots, 0);

    // Parent flag addresses are packed in stable node-index order by
    // prepareUniRunnerIpcParentFlags. Validate that layout independently of
    // the filtered topological trigger order.
    size_t expectedParentOffset = 0;
    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      const uniRunnerDagNode &node = nodes[nodeIdx];
      if (node.nodeIdx != static_cast<int>(nodeIdx)) {
        return flagcxInternalError;
      }
      if (node.nodeType != uniRunnerDagNodeTypeIpc) {
        continue;
      }
      if (node.numParents < 0 ||
          (node.numParents > 0 && node.parents == NULL) ||
          node.nodeData.ipc.parentFlagsOffset != expectedParentOffset ||
          expectedParentOffset > config->parentFlagsCount ||
          static_cast<size_t>(node.numParents) >
              config->parentFlagsCount - expectedParentOffset) {
        return flagcxInvalidArgument;
      }
      expectedParentOffset += static_cast<size_t>(node.numParents);
    }
    if (expectedParentOffset != config->parentFlagsCount) {
      return flagcxInvalidArgument;
    }

    size_t hostNodes = 0;
    size_t redNodes = 0;
    size_t ipcNodes = 0;
    size_t validatedMaxChunks = 0;
    for (size_t topoSlot = 0; topoSlot < numNodes; ++topoSlot) {
      const int nodeIdx = plan->topoOrder[topoSlot];
      if (nodeIdx < 0 || static_cast<size_t>(nodeIdx) >= numNodes ||
          seenNodes[nodeIdx] != 0 || nodes[nodeIdx].nodeIdx != nodeIdx) {
        return flagcxInternalError;
      }
      seenNodes[nodeIdx] = 1;
      const uniRunnerDagNode &node = nodes[nodeIdx];
      if (node.nodeType == uniRunnerDagNodeTypeP2p ||
          node.nodeType == uniRunnerDagNodeTypeCpy) {
        ++hostNodes;
        continue;
      }
      if (node.nodeType == uniRunnerDagNodeTypeRed) {
        ++redNodes;
        continue;
      }
      if (node.nodeType != uniRunnerDagNodeTypeIpc ||
          ipcNodes >= plan->numIpcNodes) {
        return flagcxInternalError;
      }
      ++ipcNodes;

      const uniRunnerIpcNodeData &ipc = node.nodeData.ipc;
      if (!isValidStaticIpcBufferType(ipc.srcBufferType) ||
          ipc.peerLocalRank < 0 || ipc.peerLocalRank >= config->localRanks ||
          ipc.readySlot >= config->readySlots ||
          seenReadySlots[ipc.readySlot] != 0 || nodeFlags[nodeIdx] == NULL ||
          ipc.srcOffsetBytes > config->dataBytes ||
          ipc.bytes > config->dataBytes - ipc.srcOffsetBytes ||
          ipc.dstOffsetBytes > config->dataBytes ||
          ipc.bytes > config->dataBytes - ipc.dstOffsetBytes) {
        return flagcxInvalidArgument;
      }
      seenReadySlots[ipc.readySlot] = 1;

      for (int parentSlot = 0; parentSlot < node.numParents; ++parentSlot) {
        const int parentIdx = node.parents[parentSlot];
        if (parentIdx < 0 || static_cast<size_t>(parentIdx) >= numNodes ||
            nodeFlags[parentIdx] == NULL) {
          return flagcxInvalidArgument;
        }
      }

      uint32_t chunks = 0;
      flagcxResult_t chunkResult = checkedUniRunnerIpcChunkCount(
          ipc.bytes, config->chunkSize, &chunks);
      if (chunkResult != flagcxSuccess) {
        return chunkResult;
      }
      validatedMaxChunks =
          std::max(validatedMaxChunks, static_cast<size_t>(chunks));
    }
    if (hostNodes != plan->numHostNodes || redNodes != plan->numRedNodes ||
        ipcNodes != plan->numIpcNodes) {
      return flagcxInternalError;
    }
    for (size_t readySlot = 0; readySlot < config->readySlots; ++readySlot) {
      if (seenReadySlots[readySlot] == 0) {
        return flagcxInvalidArgument;
      }
    }

    // The validation pass above is intentionally complete: no trigger or DAG
    // index is published until every entry can be materialized successfully.
    size_t ipcOrdinal = 0;
    for (size_t topoSlot = 0; topoSlot < numNodes; ++topoSlot) {
      const int nodeIdx = plan->topoOrder[topoSlot];
      uniRunnerDagNode &node = nodes[nodeIdx];
      if (node.nodeType != uniRunnerDagNodeTypeIpc) {
        continue;
      }
      const uniRunnerIpcNodeData &ipc = node.nodeData.ipc;
      uint32_t chunks = 0;
      (void)checkedUniRunnerIpcChunkCount(ipc.bytes, config->chunkSize,
                                         &chunks);
      flagcxIpcTrigger &trigger = triggers[ipcOrdinal];
      trigger.srcOffsetBytes = ipc.srcOffsetBytes;
      trigger.dstOffsetBytes = ipc.dstOffsetBytes;
      trigger.bytes = ipc.bytes;
      trigger.chunkSize = config->chunkSize;
      trigger.flagOut = reinterpret_cast<uint64_t>(nodeFlags[nodeIdx]);
      trigger.epoch = config->epoch;
      trigger.srcBufferType = static_cast<uint32_t>(ipc.srcBufferType);
      trigger.peerLocalRank = static_cast<uint32_t>(ipc.peerLocalRank);
      trigger.readySlot = ipc.readySlot;
      trigger.parentFlagsOffset = ipc.parentFlagsOffset;
      trigger.numParentFlags = static_cast<uint32_t>(node.numParents);
      trigger.numChunks = chunks;
      trigger.completedChunks = 0;
      trigger.nextChunk = 0;
      trigger.state = flagcxReduceTriggerEnqueued;
      node.nodeData.ipc.triggerIdx = static_cast<int>(ipcOrdinal++);
    }
    __sync_synchronize();
    *numTriggers = ipcOrdinal;
    *maxChunksPerTrigger = validatedMaxChunks;
    return flagcxSuccess;
  } catch (...) {
    return flagcxSystemError;
  }
}
