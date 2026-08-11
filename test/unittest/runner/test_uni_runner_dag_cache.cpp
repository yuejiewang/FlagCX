#include <cstdint>
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
