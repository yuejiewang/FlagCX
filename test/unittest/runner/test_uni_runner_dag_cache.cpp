#include <cstdint>
#include <limits>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "uni_runner_helper.h"

namespace {

uniRunnerDagCacheKey makeCacheKey(uint64_t algoHash) {
  uniRunnerDagCacheKey key{};
  key.algoType = uniRunnerDagAlgoSlicedAR;
  key.algoHash = algoHash;
  key.commOp = flagcxCommOpAllReduce;
  key.count = 1048576;
  key.datatype = flagcxFloat32;
  key.redOp = flagcxSum;
  key.rank = 3;
  key.nranks = 8;
  key.root = -1;
  return key;
}

uniRunnerDagTemplate makeIpcDagTemplate() {
  uniRunnerDagTemplate dagTemplate;
  dagTemplate.key = makeCacheKey(0x123456789abcdef0ull);
  dagTemplate.key.algoType = uniRunnerDagAlgoIpcAR;

  uniRunnerDagNodeDesc node;
  node.nodeType = uniRunnerDagNodeTypeIpc;
  node.nodeIdx = 0;
  node.ipc.srcOffsetBytes = 128;
  node.ipc.dstOffsetBytes = 256;
  node.ipc.bytes = 4096;
  node.ipc.srcBufferType = flagcxIpcBufferOutput;
  node.ipc.peerLocalRank = 7;
  node.ipc.readySlot = 3;
  dagTemplate.nodes.push_back(node);
  return dagTemplate;
}

} // namespace

TEST(UniRunnerDagCache, AlgorithmHashIsStableAndCoversBuilderInputs) {
  uniRunnerDagAlgorithmConfig config{};
  config.numSlices = 2;
  config.numRedSlices = 8;
  config.groupSize = 4;
  config.topologyHash = 0x1020304050607080ull;
  config.bufferMode = uniRunnerDagBufferModeOutOfPlace;

  const uint64_t expected =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, config);
  // Golden FNV-1a value protects the persisted identity from accidental
  // dependence on process or standard-library hashing behavior.
  EXPECT_EQ(6300673621123503851ull, expected);
  EXPECT_EQ(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, config));

  uniRunnerDagAlgorithmConfig changed = config;
  changed.numSlices++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));
  changed = config;
  changed.numRedSlices++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));
  changed = config;
  changed.bufferMode = uniRunnerDagBufferModeInPlace;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, changed));

  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoRingAR, config));
}

TEST(UniRunnerDagCache, AlgorithmHashIgnoresUnrelatedBuilderSlots) {
  uniRunnerDagAlgorithmConfig sliced{};
  sliced.numSlices = 2;
  sliced.numRedSlices = 8;
  const uint64_t slicedHash =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, sliced);
  sliced.groupSize = 4;
  sliced.topologyHash = 0x1234;
  EXPECT_EQ(slicedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoSlicedAR, sliced));

  uniRunnerDagAlgorithmConfig grouped{};
  grouped.groupSize = 4;
  const uint64_t groupedHash =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped);
  grouped.numSlices = 2;
  grouped.numRedSlices = 8;
  grouped.topologyHash = 0x1234;
  EXPECT_EQ(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));

  grouped = {};
  grouped.groupSize = 8;
  EXPECT_NE(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));
  grouped.groupSize = 4;
  grouped.bufferMode = uniRunnerDagBufferModeInPlace;
  EXPECT_NE(groupedHash,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoGroupedAG, grouped));
}

TEST(UniRunnerDagCache, IpcAlgorithmHashIncludesLocalTopology) {
  uniRunnerDagAlgorithmConfig config{};
  config.numSlices = 2;
  config.numRedSlices = 8;
  config.topologyHash = 0x1234;
  const uint64_t expected =
      getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoIpcAR, config);
  config.topologyHash++;
  EXPECT_NE(expected,
            getUniRunnerDagAlgorithmHash(uniRunnerDagAlgoIpcAR, config));
}

TEST(UniRunnerDagCache, PatternHashIncludesAlgorithmIdentity) {
  const uniRunnerDagCacheKey first = makeCacheKey(0x123456789abcdef0ull);
  uniRunnerDagCacheKey second = first;
  second.algoHash++;
  EXPECT_NE(getUniRunnerDagPatternHash(first),
            getUniRunnerDagPatternHash(second));

  second = first;
  second.algoType = uniRunnerDagAlgoRingAR;
  EXPECT_NE(getUniRunnerDagPatternHash(first),
            getUniRunnerDagPatternHash(second));
}

TEST(UniRunnerDagCache, JsonKeyContainsOnlyAlgorithmAndCollectiveIdentity) {
  const uniRunnerDagCacheKey key = makeCacheKey(0xfedcba9876543210ull);
  const Json json = uniRunnerDagCacheKeyToJson(key);
  const std::set<std::string> expectedFields = {
      "algo_name", "algo_hash", "comm_op", "count", "datatype",
      "red_op",   "rank",      "nranks",  "root"};
  std::set<std::string> actualFields;
  for (auto it = json.begin(); it != json.end(); ++it) {
    actualFields.insert(it.key());
  }

  EXPECT_EQ(expectedFields, actualFields);
  EXPECT_EQ("sliced_ar", json.at("algo_name").get<std::string>());
  EXPECT_TRUE(json.at("algo_hash").is_string());
  EXPECT_EQ("18364758544493064720",
            json.at("algo_hash").get<std::string>());
  EXPECT_FALSE(json.contains("format_version"));
  EXPECT_FALSE(json.contains("num_slices"));
  EXPECT_FALSE(json.contains("num_red_slices"));
  EXPECT_FALSE(json.contains("group_size"));
}

TEST(UniRunnerDagCache, JsonRoundTripPreservesFullWidthAlgorithmHash) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);

  const Json json = uniRunnerSerializeDagTemplate(original);
  ASSERT_EQ(kUniRunnerDagCacheFormatVersion,
            json.at("format_version").get<int>());

  uniRunnerDagTemplate decoded;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &decoded));
  EXPECT_EQ(original.key.algoType, decoded.key.algoType);
  EXPECT_EQ(original.key.algoHash, decoded.key.algoHash);
  EXPECT_EQ(original.key.commOp, decoded.key.commOp);
  EXPECT_EQ(original.key.count, decoded.key.count);
  EXPECT_EQ(original.key.datatype, decoded.key.datatype);
  EXPECT_EQ(original.key.redOp, decoded.key.redOp);
  EXPECT_EQ(original.key.rank, decoded.key.rank);
  EXPECT_EQ(original.key.nranks, decoded.key.nranks);
  EXPECT_EQ(original.key.root, decoded.key.root);
  EXPECT_EQ(getUniRunnerDagPatternHash(original.key), decoded.hashValue);
}

TEST(UniRunnerDagCache, IpcDescriptorJsonRoundTripContainsOnlyStructure) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json json = uniRunnerSerializeDagTemplate(original);
  ASSERT_EQ(4, json.at("format_version").get<int>());
  ASSERT_EQ(1u, json.at("dag").at("nodes").size());

  const Json &nodeJson = json.at("dag").at("nodes").at(0);
  ASSERT_TRUE(nodeJson.contains("ipc"));
  const Json &ipcJson = nodeJson.at("ipc");
  const std::set<std::string> expectedFields = {
      "src_offset_bytes", "dst_offset_bytes", "bytes",
      "src_buffer_type",  "peer_local_rank", "ready_slot"};
  std::set<std::string> actualFields;
  for (auto it = ipcJson.begin(); it != ipcJson.end(); ++it) {
    actualFields.insert(it.key());
  }
  EXPECT_EQ(expectedFields, actualFields);
  EXPECT_EQ("output", ipcJson.at("src_buffer_type").get<std::string>());

  for (const char *runtimeField :
       {"parent_flags_offset", "trigger_idx", "chunk_size", "epoch",
        "num_chunks", "completed_chunks", "next_chunk", "state",
        "flag_in", "flag_out", "src_ptr", "dst_ptr", "dev_mem",
        "window"}) {
    EXPECT_FALSE(ipcJson.contains(runtimeField));
    EXPECT_FALSE(nodeJson.contains(runtimeField));
  }

  uniRunnerDagTemplate decoded;
  ASSERT_TRUE(uniRunnerDeserializeDagTemplate(json, &decoded));
  ASSERT_EQ(1u, decoded.nodes.size());
  const uniRunnerDagIpcOpDesc &ipc = decoded.nodes[0].ipc;
  EXPECT_EQ(original.nodes[0].ipc.srcOffsetBytes, ipc.srcOffsetBytes);
  EXPECT_EQ(original.nodes[0].ipc.dstOffsetBytes, ipc.dstOffsetBytes);
  EXPECT_EQ(original.nodes[0].ipc.bytes, ipc.bytes);
  EXPECT_EQ(original.nodes[0].ipc.srcBufferType, ipc.srcBufferType);
  EXPECT_EQ(original.nodes[0].ipc.peerLocalRank, ipc.peerLocalRank);
  EXPECT_EQ(original.nodes[0].ipc.readySlot, ipc.readySlot);
}

TEST(UniRunnerDagCache, RejectsMalformedIpcDescriptorFields) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json valid = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  for (const char *requiredField :
       {"src_offset_bytes", "dst_offset_bytes", "bytes",
        "src_buffer_type", "peer_local_rank", "ready_slot"}) {
    Json json = valid;
    json["dag"]["nodes"][0]["ipc"].erase(requiredField);
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded))
        << requiredField;
  }

  Json json = valid;
  json["dag"]["nodes"][0]["ipc"]["epoch"] = 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["trigger_idx"] = 0;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["src_buffer_type"] = "scratch";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["src_buffer_type"] = 0;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  for (const char *sizeField :
       {"src_offset_bytes", "dst_offset_bytes", "bytes"}) {
    json = valid;
    json["dag"]["nodes"][0]["ipc"][sizeField] = -1;
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)) << sizeField;

    json = valid;
    json["dag"]["nodes"][0]["ipc"][sizeField] = "1";
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)) << sizeField;
  }
}

TEST(UniRunnerDagCache, RejectsIpcPeerAndReadySlotOverflow) {
  const uniRunnerDagTemplate original = makeIpcDagTemplate();
  const Json valid = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  Json json = valid;
  json["dag"]["nodes"][0]["ipc"]["peer_local_rank"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["peer_local_rank"] =
      static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["ready_slot"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["dag"]["nodes"][0]["ipc"]["ready_slot"] =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = valid;
  json["format_version"] = 3;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagCache, RejectsOldOrMalformedAlgorithmIdentity) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  Json json = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  json["format_version"] = kUniRunnerDagCacheFormatVersion - 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["algo_hash"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["algo_hash"] = "123garbage";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  for (const char *invalidHash : {"+1", " 1", "18446744073709551616"}) {
    json = uniRunnerSerializeDagTemplate(original);
    json["key"]["algo_hash"] = invalidHash;
    EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
  }

  json = uniRunnerSerializeDagTemplate(original);
  json["hash"] = "0";
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagCache, MalformedDocumentsReturnFalseWithoutThrowing) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  uniRunnerDagTemplate decoded;

  Json json = uniRunnerSerializeDagTemplate(original);
  json.erase("key");
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json.erase("dag");
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json["format_version"] = "3";
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));

  json = uniRunnerSerializeDagTemplate(original);
  json["format_version"] = 18446744073709551615ull;
  EXPECT_NO_THROW(
      EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded)));
}

TEST(UniRunnerDagCache, RejectsUnsafeDagArraySizes) {
  uniRunnerDagTemplate original;
  original.key = makeCacheKey(0xfedcba9876543210ull);
  Json json = uniRunnerSerializeDagTemplate(original);
  uniRunnerDagTemplate decoded;

  json["dag"]["num_nodes"] = 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["num_nodes"] =
      static_cast<uint64_t>(std::numeric_limits<int>::max()) + 1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["num_nodes"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["dag"]["nodes"] = Json::object();
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));

  json = uniRunnerSerializeDagTemplate(original);
  json["key"]["count"] = -1;
  EXPECT_FALSE(uniRunnerDeserializeDagTemplate(json, &decoded));
}

TEST(UniRunnerDagSizing, CheckedNodeCountRejectsIntOverflow) {
  int nodeCount = -1;
  const size_t maxInt =
      static_cast<size_t>(std::numeric_limits<int>::max());

  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(0, maxInt, 0, &nodeCount));
  EXPECT_EQ(0, nodeCount);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(1, maxInt, 0, &nodeCount));
  EXPECT_EQ(std::numeric_limits<int>::max(), nodeCount);
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerDagNodeCount(maxInt - 1, 1, 1, &nodeCount));
  EXPECT_EQ(std::numeric_limits<int>::max(), nodeCount);

  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(2, maxInt, 0, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, maxInt, 1, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, 0, maxInt + 1, &nodeCount));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerDagNodeCount(1, 1, 0, nullptr));
}

TEST(UniRunnerDagSizing, CheckedTypeBytesRejectsSizeOverflow) {
  size_t bytes = 0;
  EXPECT_EQ(flagcxSuccess,
            checkedUniRunnerTypeBytes(1024, 8, flagcxInt8, &bytes));
  EXPECT_EQ(8192u, bytes);
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(std::numeric_limits<size_t>::max(), 2,
                                      flagcxInt8, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(std::numeric_limits<size_t>::max(), 1,
                                      flagcxFloat32, &bytes));
  EXPECT_EQ(flagcxInvalidArgument,
            checkedUniRunnerTypeBytes(1, 1, flagcxFloat32, nullptr));

  size_t decodedSize = 0;
  EXPECT_FALSE(uniRunnerDagSizeFromJson(Json(-1), &decodedSize));
  EXPECT_TRUE(uniRunnerDagSizeFromJson(Json(0), &decodedSize));
  EXPECT_EQ(0u, decodedSize);
}
