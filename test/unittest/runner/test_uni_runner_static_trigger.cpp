#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "uni_runner_impl.h"

namespace {

void *fakePtr(uintptr_t value) { return reinterpret_cast<void *>(value); }

void setValidRedPayload(uniRunnerDagNode *node, uintptr_t base) {
  node->nodeData.red.input1 = fakePtr(base + 0x100);
  node->nodeData.red.input2 = fakePtr(base + 0x200);
  node->nodeData.red.output = fakePtr(base + 0x300);
  node->nodeData.red.count = 17;
  node->nodeData.red.nthreads = 32;
  node->nodeData.red.datatype = flagcxFloat32;
  node->nodeData.red.redOp = flagcxSum;
}

} // namespace

TEST(UniRunnerStaticTrigger, ChecksHeaderPlusPayloadAllocationSize) {
  size_t bytes = 0;
  const size_t headerBytes = flagcxFifoIdxData * sizeof(uint64_t);
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(
                129, sizeof(flagcxReduceTrigger), &bytes));
  EXPECT_EQ(headerBytes + 129 * sizeof(flagcxReduceTrigger), bytes);

  const size_t maxCapacity =
      static_cast<size_t>(std::numeric_limits<int>::max());
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(
                maxCapacity, sizeof(flagcxReduceTrigger), &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(
                maxCapacity + 1, sizeof(flagcxReduceTrigger), &bytes));
  const size_t maxElementSize =
      std::numeric_limits<size_t>::max() - headerBytes;
  EXPECT_EQ(flagcxSuccess,
            flagcxCheckedFifoAllocationSize(1, maxElementSize, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, maxElementSize + 1, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, 0, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            flagcxCheckedFifoAllocationSize(1, 1, nullptr));
}

TEST(UniRunnerStaticTrigger, MaterializesRedNodesInFilteredTopoOrder) {
  std::vector<uniRunnerDagNode> nodes(3);
  for (size_t i = 0; i < nodes.size(); ++i) {
    nodes[i] = {};
    nodes[i].nodeIdx = static_cast<int>(i);
  }
  int parentOfZero = 1;
  int parentOfOne = 2;
  nodes[0].nodeType = uniRunnerDagNodeTypeRed;
  nodes[0].numParents = 1;
  nodes[0].parents = &parentOfZero;
  nodes[0].nodeData.red.input1 = fakePtr(0x1000);
  nodes[0].nodeData.red.input2 = fakePtr(0x2000);
  nodes[0].nodeData.red.output = fakePtr(0x3000);
  nodes[0].nodeData.red.count = 17;
  nodes[0].nodeData.red.nthreads = 32;
  nodes[0].nodeData.red.datatype = flagcxFloat32;
  nodes[0].nodeData.red.redOp = flagcxSum;
  nodes[0].nodeData.red.triggerIdx = -1;

  nodes[1].nodeType = uniRunnerDagNodeTypeRed;
  nodes[1].numParents = 1;
  nodes[1].parents = &parentOfOne;
  nodes[1].nodeData.red.input1 = fakePtr(0x4000);
  nodes[1].nodeData.red.input2 = fakePtr(0x5000);
  nodes[1].nodeData.red.output = fakePtr(0x6000);
  nodes[1].nodeData.red.count = 19;
  nodes[1].nodeData.red.nthreads = 32;
  nodes[1].nodeData.red.datatype = flagcxFloat32;
  nodes[1].nodeData.red.redOp = flagcxMax;
  nodes[1].nodeData.red.triggerIdx = -1;

  nodes[2].nodeType = uniRunnerDagNodeTypeP2p;
  const int topoOrder[] = {2, 1, 0};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = const_cast<int *>(topoOrder);
  plan.numNodes = 3;
  plan.numHostNodes = 1;
  plan.numRedNodes = 2;
  void *flags[] = {fakePtr(0x7000), fakePtr(0x8000), fakePtr(0x9000)};
  flagcxReduceTrigger triggers[2] = {};
  size_t numTriggers = 0;

  ASSERT_EQ(flagcxSuccess, populateUniRunnerStaticRedTriggers(
                               nodes.data(), nodes.size(), &plan, flags, 3,
                               triggers, 2, &numTriggers));
  ASSERT_EQ(2u, numTriggers);
  EXPECT_EQ(0, nodes[1].nodeData.red.triggerIdx);
  EXPECT_EQ(1, nodes[0].nodeData.red.triggerIdx);
  EXPECT_EQ(reinterpret_cast<uint64_t>(nodes[1].nodeData.red.input1),
            triggers[0].value[0]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(nodes[0].nodeData.red.input1),
            triggers[1].value[0]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[2]), triggers[0].value[4]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[1]), triggers[1].value[4]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[1]), triggers[0].value[5]);
  EXPECT_EQ(reinterpret_cast<uint64_t>(flags[0]), triggers[1].value[5]);
  EXPECT_EQ(flagcxReduceTriggerEnqueued, triggers[0].pollState());
  EXPECT_EQ(flagcxReduceTriggerEnqueued, triggers[1].pollState());
}

TEST(UniRunnerStaticTrigger, DoesNotPublishPartialMaterialization) {
  uniRunnerDagNode node = {};
  node.nodeIdx = 0;
  node.nodeType = uniRunnerDagNodeTypeRed;
  node.nodeData.red.triggerIdx = -1;
  setValidRedPayload(&node, 0x1000);
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = &topoOrder;
  plan.numNodes = 1;
  plan.numRedNodes = 1;
  void *flag = fakePtr(0x1000);
  flagcxReduceTrigger trigger = {};
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          0, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  int parents[] = {0, 0};
  node.numParents = 2;
  node.parents = parents;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  node.numParents = 0;
  node.parents = nullptr;
  node.nodeData.red.count =
      flagcxTriggerMask(flagcxReduceTriggerBitsCount) + 1;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);

  node.nodeData.red.count = 17;
  node.nodeData.red.nthreads = 0;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.nthreads =
      flagcxTriggerMask(flagcxReduceTriggerBitsNThreads) + 1;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.nthreads = 32;
  node.nodeData.red.datatype = static_cast<flagcxDataType_t>(flagcxNumTypes);
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  node.nodeData.red.datatype = flagcxFloat32;
  node.nodeData.red.redOp = flagcxNumRedOps;
  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          &node, 1, &plan, &flag, 1, &trigger,
                                          1, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-1, node.nodeData.red.triggerIdx);
}

TEST(UniRunnerStaticTrigger, ValidatesAllNodesBeforeWritingAnyTrigger) {
  uniRunnerDagNode nodes[2] = {};
  for (int i = 0; i < 2; ++i) {
    nodes[i].nodeIdx = i;
    nodes[i].nodeType = uniRunnerDagNodeTypeRed;
    nodes[i].nodeData.red.triggerIdx = -7 - i;
    setValidRedPayload(&nodes[i], 0x1000 * static_cast<uintptr_t>(i + 1));
  }
  nodes[1].nodeData.red.nthreads = 0;
  int topoOrder[] = {0, 1};
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = topoOrder;
  plan.numNodes = 2;
  plan.numRedNodes = 2;
  void *flags[] = {fakePtr(0x7000), fakePtr(0x8000)};
  flagcxReduceTrigger triggers[2] = {};
  triggers[0].value[0] = 0xaaaa;
  triggers[1].value[0] = 0xbbbb;
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxInvalidArgument, populateUniRunnerStaticRedTriggers(
                                          nodes, 2, &plan, flags, 2, triggers,
                                          2, &numTriggers));
  EXPECT_EQ(0u, numTriggers);
  EXPECT_EQ(-7, nodes[0].nodeData.red.triggerIdx);
  EXPECT_EQ(-8, nodes[1].nodeData.red.triggerIdx);
  EXPECT_EQ(0xaaaau, triggers[0].value[0]);
  EXPECT_EQ(0xbbbbu, triggers[1].value[0]);
}

TEST(UniRunnerStaticTrigger, AcceptsHostOnlyPlanWithoutTriggerStorage) {
  uniRunnerDagNode node = {};
  node.nodeIdx = 0;
  node.nodeType = uniRunnerDagNodeTypeP2p;
  int topoOrder = 0;
  uniRunnerDagExecutionPlan plan = {};
  plan.topoOrder = &topoOrder;
  plan.numNodes = 1;
  plan.numHostNodes = 1;
  size_t numTriggers = 99;

  EXPECT_EQ(flagcxSuccess, populateUniRunnerStaticRedTriggers(
                               &node, 1, &plan, nullptr, 0, nullptr, 0,
                               &numTriggers));
  EXPECT_EQ(0u, numTriggers);
}
