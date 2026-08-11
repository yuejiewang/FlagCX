/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "uni_runner_impl.h"

#include "alloc.h"

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
