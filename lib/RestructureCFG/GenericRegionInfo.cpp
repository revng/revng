//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/Analysis/CycleAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"

#include "revng/RestructureCFG/GenericRegionInfo.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"

using namespace llvm;

// Debug logger
Logger<> GenericRegionInfoLogger("generic-region-info");

/// Helper function which mimics the `at` behavior for a `llvm::SmallDenseMap`
template<class KeyT, class ValueT>
static ValueT mapAt(llvm::SmallDenseMap<KeyT, ValueT> &Map, KeyT Key) {
  auto MapIt = Map.find(Key);
  revng_assert(MapIt != Map.end());
  return MapIt->second;
}

template<class GraphT, class GT>
void GenericRegionInfo<GraphT, GT>::initializeRegions(GraphT F) {
  // We instantiate the `GenericCycle` analysis and wrap the results in
  // the region objects
  CycleInfo CI;
  CI.compute(*F);

  using CycleT = CycleInfo::CycleT;
  using Region = GenericRegion<NodeT>;
  llvm::SmallDenseMap<const CycleT *, Region *> CycleToRegionMap;

  // Populate the `Regions` with the identified regions
  for (const auto *TLC : CI.toplevel_cycles()) {
    for (const auto *Cycle : depth_first(TLC)) {

      // Create a new `Region`
      Regions.push_back(std::make_unique<Region>());
      Region *CurrentRegion = Regions.back().get();

      // Populate the mapping between the `CycleT` object and our custom
      // `Region`
      CycleToRegionMap[Cycle] = CurrentRegion;

      // Iterate over all the blocks and insert them in the `CurrentRegion`
      for (auto *Block : Cycle->blocks()) {
        CurrentRegion->insertBlock(Block);
      }
    }
  }

  // Populate the children regions. We need to perform this operation in a
  // separate step in order to have already all the created regions in the step
  // above
  for (const auto *TLC : CI.toplevel_cycles()) {
    for (const auto *Cycle : depth_first(TLC)) {
      auto *Region = mapAt(CycleToRegionMap, Cycle);
      for (const auto *Child : Cycle->children()) {
        auto *ChildRegion = mapAt(CycleToRegionMap, Child);
        Region->addChild(ChildRegion);
      }
    }
  }
}

/// Helper static function to compute the shortest distance from the entry block
template<class GraphT>
static llvm::SmallDenseMap<typename llvm::GraphTraits<GraphT>::NodeRef, size_t>
computeShortesPath(GraphT F) {
  using NodeT = llvm::GraphTraits<GraphT>::NodeRef;
  llvm::SmallDenseMap<NodeT, size_t> ShortestPathFromEntry;

  for (auto BFSIt = bf_begin(F); BFSIt != bf_end(F); BFSIt++) {
    NodeT Block = *BFSIt;
    size_t Depth = BFSIt.getLevel();

    // Obtain the insertion iterator for the `Depth` block element
    auto ShortestIt = ShortestPathFromEntry.insert({ Block, Depth });

    // If we already had in the map an entry for the current block, we need to
    // assert that the previously found value for the `Depth` is less or equal
    // of the `Depth` we are inserting
    if (ShortestIt.second == false) {
      revng_assert(ShortestIt.first->second <= Depth);
    }
  }

  return ShortestPathFromEntry;
}

template<class GraphT, class GT>
void GenericRegionInfo<GraphT, GT>::electHead(GraphT F) {

  // For each `Region`, we perform the election of the head node
  // 1) Compute the reverse post order
  llvm::SmallVector<NodeT> RPOT;
  llvm::copy(llvm::post_order(F), std::back_inserter(RPOT));

  // 2) We use the shortest distance from the entry block of the function just
  //    as a tie breaker. Therefore, we delay its computation until it is
  //    necessary.
  std::optional<llvm::SmallDenseMap<NodeT, size_t>>
    ShortestPathFromEntry = std::nullopt;

  // 3) Perform the election for each `Region`
  for (auto &CurrentRegion : Regions) {

    // All the blocks which have an incoming edge from a block not part of the
    // region itself, are considered as head candidates
    llvm::SmallMapVector<NodeT, size_t, 4> HeadCandidates;
    for (NodeT Block : CurrentRegion->blocks()) {
      for (NodeT Predecessor : graph_predecessors(Block)) {
        if (not CurrentRegion->containsBlock(Predecessor)) {
          HeadCandidates[Block]++;
        }
      }
    }

    // Elect the `Head` as the candidate head with the largest number of
    // incoming edges from outside the region.
    // If there is a tie, i.e., there are 2 or more candidate heads with the
    // same number of incoming edges from outside the region itself, we select
    // the entry with the minimal shortest path from entry. If it is still a
    // tie, i.e., there are 2 or more candidate heads with, also, the same
    // minimal shortest path from entry, then we disambiguate by picking the
    // head that comes first in RPOT.
    NodeT Head = HeadCandidates.begin()->first;
    {
      size_t MaxNHead = HeadCandidates.begin()->second;
      auto HeadEnd = HeadCandidates.end();
      for (NodeT Block : RPOT) {
        auto HeadIt = HeadCandidates.find(Block);
        if (HeadIt != HeadEnd) {
          const auto &[HeadCandidate, NumIncoming] = *HeadIt;
          if (NumIncoming > MaxNHead) {
            Head = HeadCandidate;
          } else if (NumIncoming == MaxNHead) {

            // Compute the `ShortestPathFromEntry` map since we need to break
            // a tie here
            if (not ShortestPathFromEntry.has_value()) {
              ShortestPathFromEntry = computeShortesPath(F);
            }
            size_t CurrentShortest = mapAt(*ShortestPathFromEntry, Head);
            size_t CandidateShortest = mapAt(*ShortestPathFromEntry,
                                             HeadCandidate);
            if (CandidateShortest < CurrentShortest) {
              Head = HeadCandidate;
            }
          }
        }
      }
    }

    // Verify that we found a `Head`
    revng_assert(Head != nullptr);

    // Set the `Head` for the `Region`
    CurrentRegion->setHead(Head);
  }
}

template<class GraphT, class GT>
void GenericRegionInfo<GraphT, GT>::compute(GraphT F) {

  initializeRegions(F);
  electHead(F);

  // Print the `GenericRegionInfo` results, when the respective Logger is
  // activated. This is used both for debugging purposes and for testing with
  // `FileCheck`.
  revng_log(GenericRegionInfoLogger, print());
}

template<class GraphT, class GT>
std::string GenericRegionInfo<GraphT, GT>::print() const {
  std::string Output;

  // Print each top level `GenericRegion`, and for each one explore it using a
  // DFS strategy
  size_t RegionIndex = 0;
  Output += "\nGeneric Region Info Results:\n";
  for (auto &TopLevelRegion : top_level_regions()) {
    for (auto *Region : llvm::depth_first(&TopLevelRegion)) {
      Output += "\nRegion " + std::to_string(RegionIndex) + ":\n";
      Output += "Elected head: " + Region->getHead()->getName().str() + "\n";
      for (auto &Block : Region->blocks()) {
        Output += Block->getName().str() + "\n";
      }
      RegionIndex++;
    }
  }

  return Output;
}

template class GenericRegionInfo<llvm::Function *>;
