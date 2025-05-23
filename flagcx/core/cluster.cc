#include "cluster.h"
#include <cstring>

flagcxResult_t flagcxCollectClusterInfos(const flagcxVendor *allData,
                                         flagcxCommunicatorType_t *type,
                                         int *homo_rank, int *homo_root_rank,
                                         int *homo_ranks, int *cluster_id,
                                         int *cluster_inter_rank, int *ncluster,
                                         int rank, int nranks) {
  *homo_rank = rank;
  *homo_root_rank = 0;
  *homo_ranks = 1;
  *cluster_id = 0;
  *cluster_inter_rank = -1;
  *ncluster = 1;
  *type = flagcxCommunicatorHomo;

  if (nranks <= 1)
    return flagcxSuccess;

  std::map<std::string, int> clusterMap;
  clusterMap[allData[0].internal] = 1;
  int numClusters = 1;
  int currCluster = 0;
  int aggRanks = 1;
  int homoRootRank = 0;
  std::string myCls = allData[rank].internal;
  for (int i = 1; i < nranks; ++i) {
    std::string cls = allData[i].internal;
    auto it = clusterMap.find(cls);
    if (it != clusterMap.end()) {
      it->second = it->second + 1;
    } else {
      clusterMap[cls] = 1;
      numClusters += 1;
      if (myCls == cls) {
        *homo_rank = *homo_rank - aggRanks;
        currCluster = numClusters - 1;
        homoRootRank = i;
      }
    }
    aggRanks += 1;

    if (i == rank) {
      *homo_root_rank = homoRootRank;
    }
  }

  *homo_ranks = clusterMap[myCls];

  if (clusterMap.size() > 1) {
    *type = flagcxCommunicatorHybrid;
  } else {
    *type = flagcxCommunicatorHomo;
  }

  if (*type == flagcxCommunicatorHybrid) {
    const char *useDev = flagcxGetEnv("FLAGCX_USEDEV");
    int useDev_;
    if (useDev == NULL) {
      useDev_ = -1;
    } else {
      useDev_ = std::stoi(useDev);
    }
    if (*homo_rank == useDev_) {
      *cluster_inter_rank = rank;
    }
    if (*homo_ranks <= useDev_ && *homo_rank == *homo_ranks - 1) {
      *cluster_inter_rank = rank;
    }
    *cluster_id = currCluster;
    *ncluster = numClusters;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxFillClusterVendorInfo(const flagcxVendor *allData,
                                           flagcxComm *comm, int *clusterIdData,
                                           int nranks, int ncluster) {
  comm->clusterVendorMap.resize(ncluster);
  for (int i = 0; i < nranks; i++) {
    std::string vendor = allData[i].internal;
    int cluster = clusterIdData[i];
    if (vendor == "NVIDIA") {
      comm->clusterVendorMap[cluster] = FLAGCX_VENDOR_NVIDIA;
    } else if (vendor == "ILUVATAR_COREX") {
      comm->clusterVendorMap[cluster] = FLAGCX_VENDOR_ILUVATAR_COREX;
    } else if (vendor == "MLU") {
      comm->clusterVendorMap[cluster] = FLAGCX_VENDOR_MLU;
    } else if (vendor == "METAX") {
      comm->clusterVendorMap[cluster] = FLAGCX_VENDOR_METAX;
    }
  }
  return flagcxSuccess;
}