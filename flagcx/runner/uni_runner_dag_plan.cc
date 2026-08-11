/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "uni_runner_impl.h"

#include "alloc.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

enum uniRunnerDagPlanPhase {
  uniRunnerDagPlanPhaseHost = 0,
  uniRunnerDagPlanPhaseRed = 1,
  uniRunnerDagPlanPhaseIpc = 2,
  uniRunnerDagPlanNumPhases = 3
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
    default:
      return false;
  }
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

} // namespace

void destroyUniRunnerDagExecutionPlan(uniRunnerDagExecutionPlan *plan) {
  if (plan == NULL) {
    return;
  }
  if (plan->topoOrder != NULL) {
    free(plan->topoOrder);
  }
  plan->topoOrder = NULL;
  plan->numNodes = 0;
  plan->numHostNodes = 0;
  plan->numRedNodes = 0;
  plan->numIpcNodes = 0;
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

  try {
    std::vector<int> indegree(numNodes, 0);
    std::vector<int> nextReady(numNodes, -1);
    std::array<int, uniRunnerDagPlanNumPhases> readyHead = {{-1, -1, -1}};
    std::array<int, uniRunnerDagPlanNumPhases> readyTail = {{-1, -1, -1}};

    const auto enqueueReady = [&](int phase, int nodeIdx) {
      if (readyTail[phase] == -1) {
        readyHead[phase] = nodeIdx;
      } else {
        nextReady[readyTail[phase]] = nodeIdx;
      }
      readyTail[phase] = nodeIdx;
    };

    for (size_t nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
      const uniRunnerDagNode &node = nodes[nodeIdx];
      int phase = -1;
      if (node.nodeIdx != static_cast<int>(nodeIdx) ||
          node.numParents < 0 || node.numChildren < 0 ||
          (node.numParents > 0 && node.parents == NULL) ||
          (node.numChildren > 0 && node.children == NULL) ||
          !getUniRunnerDagPlanPhase(node.nodeType, &phase)) {
        return flagcxInternalError;
      }

      for (int parentSlot = 0; parentSlot < node.numParents; ++parentSlot) {
        const int parentIdx = node.parents[parentSlot];
        if (parentIdx < 0 || static_cast<size_t>(parentIdx) >= numNodes ||
            parentIdx == static_cast<int>(nodeIdx)) {
          return flagcxInternalError;
        }
      }
      for (int childSlot = 0; childSlot < node.numChildren; ++childSlot) {
        const int childIdx = node.children[childSlot];
        if (childIdx < 0 || static_cast<size_t>(childIdx) >= numNodes ||
            childIdx == static_cast<int>(nodeIdx)) {
          return flagcxInternalError;
        }
      }

      indegree[nodeIdx] = node.numParents;
      if (node.numParents == 0) {
        enqueueReady(phase, static_cast<int>(nodeIdx));
      }
    }

    int *topoOrder = NULL;
    flagcxResult_t allocResult = flagcxCalloc(&topoOrder, numNodes);
    if (allocResult != flagcxSuccess) {
      return allocResult;
    }

    size_t numOrdered = 0;
    size_t numHostNodes = 0;
    size_t numRedNodes = 0;
    size_t numIpcNodes = 0;

    while (numOrdered < numNodes) {
      const size_t orderedBeforeRound = numOrdered;
      for (int phase = 0; phase < uniRunnerDagPlanNumPhases; ++phase) {
        while (readyHead[phase] != -1) {
          const int nodeIdx = readyHead[phase];
          readyHead[phase] = nextReady[nodeIdx];
          nextReady[nodeIdx] = -1;
          if (readyHead[phase] == -1) {
            readyTail[phase] = -1;
          }

          if (numOrdered >= numNodes) {
            free(topoOrder);
            return flagcxInternalError;
          }
          topoOrder[numOrdered++] = nodeIdx;
          if (phase == uniRunnerDagPlanPhaseHost) {
            ++numHostNodes;
          } else if (phase == uniRunnerDagPlanPhaseRed) {
            ++numRedNodes;
          } else {
            ++numIpcNodes;
          }

          const uniRunnerDagNode &node = nodes[nodeIdx];
          for (int childSlot = 0; childSlot < node.numChildren; ++childSlot) {
            const int childIdx = node.children[childSlot];
            if (indegree[childIdx] <= 0) {
              free(topoOrder);
              return flagcxInternalError;
            }
            --indegree[childIdx];
            if (indegree[childIdx] == 0) {
              int childPhase = -1;
              if (!getUniRunnerDagPlanPhase(nodes[childIdx].nodeType,
                                            &childPhase)) {
                free(topoOrder);
                return flagcxInternalError;
              }
              enqueueReady(childPhase, childIdx);
            }
          }
        }
      }

      if (numOrdered == orderedBeforeRound) {
        free(topoOrder);
        return flagcxInvalidArgument;
      }
    }

    if (numHostNodes + numRedNodes + numIpcNodes != numNodes) {
      free(topoOrder);
      return flagcxInternalError;
    }

    plan->topoOrder = topoOrder;
    plan->numNodes = numNodes;
    plan->numHostNodes = numHostNodes;
    plan->numRedNodes = numRedNodes;
    plan->numIpcNodes = numIpcNodes;
    return flagcxSuccess;
  } catch (...) {
    return flagcxSystemError;
  }
}

flagcxResult_t resolveUniRunnerStaticExecutorSchedule(
    const uniRunnerDagExecutionPlan *plan, size_t requestedRedBlocks,
    size_t requestedIpcBlocks, size_t maxExecutorBlocks,
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
      plan->numIpcNodes !=
          plan->numNodes - plan->numHostNodes - plan->numRedNodes) {
    return flagcxInvalidArgument;
  }

  schedule->numRedTasks = plan->numRedNodes;
  schedule->numIpcTasks = plan->numIpcNodes;
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
      std::min(requestedIpcBlocks, plan->numIpcNodes);
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
      plan->numIpcNodes !=
          numNodes - plan->numHostNodes - plan->numRedNodes) {
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
