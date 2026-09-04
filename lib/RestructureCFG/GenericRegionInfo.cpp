//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GenericCycleImpl.h"
#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/SSAContext.h"

#include "revng/RestructureCFG/GenericRegion.h"
#include "revng/RestructureCFG/GenericRegionInfo.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/GraphAlgorithms.h"

using namespace llvm;

// Debug logger
Logger GenericRegionInfoLogger("generic-region-info");

/// Helper function which mimics the `at` behavior for a `llvm::SmallDenseMap`
template<class KeyT, class ValueT>
static ValueT mapAt(llvm::SmallDenseMap<KeyT, ValueT> &Map, KeyT Key) {
  auto MapIt = Map.find(Key);
  revng_assert(MapIt != Map.end());
  return MapIt->second;
}

/// Helper function to obtain a `GenericCycleInfo` analysis
template<class GraphT>
static GenericCycleInfo<SSAContext, GraphT> getGenericCycleInfo(GraphT &F) {
  // We instantiate the `GenericCycle` analysis and wrap the results in
  // the region objects
  GenericCycleInfo<SSAContext, GraphT> GCI;
  GCI.compute(*F);

  return GCI;
}

/// Template function specialization to obtain the `GenericCycleInfo` analysis
/// starting from a `Scope<llvm::Function *>` parameter, since we need to unwrap
/// the `Graph` object from the `Scope` wrapper class
template<>
GenericCycleInfo<SSAContext, Scope<llvm::Function *>>
getGenericCycleInfo(Scope<llvm::Function *> &SG) {
  // We instantiate the `GenericCycle` analysis and wrap the results in
  // the region objects
  GenericCycleInfo<SSAContext, Scope<llvm::Function *>> GCI;
  GCI.compute(*SG.Graph);

  return GCI;
}

template<class GraphT, class GT>
void GenericRegionInfo<GraphT, GT>::initializeRegions(GraphT F) {

  // Obtain the `GenericCycleInfo` analysis
  auto GCI = getGenericCycleInfo(F);

  using CycleT = GenericCycleInfo<SSAContext, GraphT>::CycleT;
  using Region = GenericRegion<NodeT>;
  llvm::SmallDenseMap<const CycleT *, Region *> CycleToRegionMap;

  // Populate the `Regions` with the identified regions
  for (const auto *TLC : GCI.toplevel_cycles()) {
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
  for (const auto *TLC : GCI.toplevel_cycles()) {
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

/// Helper static function which computes the `Head` candidates for a given
/// region
template<class NodeT>
static llvm::SmallVector<NodeT>
getHeadCandidates(GenericRegion<NodeT> &Region) {
  llvm::SmallVector<NodeT> HeadCandidates;
  for (NodeT Block : Region.blocks()) {
    for (NodeT Predecessor : graph_predecessors(Block)) {
      if (not Region.containsBlock(Predecessor)) {
        HeadCandidates.push_back(Block);
      }
    }
  }

  return HeadCandidates;
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

  // 3) Perform the head election for each `Region`, in a bottom up fashion
  for (auto &TopLevelRegion : top_level_regions()) {
    for (auto &CurrentRegion : post_order(&TopLevelRegion)) {

      // The `Head` election phase works in a bottom-up fashion and it must
      // guarantee that the decision we take when processing a region, is
      // coherent with all the children region it contains. Specifically, we
      // have these 2 criteria:
      // 1) We must be coherent in terms of _late entries_. This means that if a
      // node is considered a late entry for a child region, it must be a late
      // entry too for its parent region. If this is not true we may end up
      // disconnecting portion of the graph from the entry. In practical terms,
      // this means that when electing the `Head` of a region, we must exclude
      // from the candidates all the nodes that happens to be late entry for its
      // children regions.
      // 2) If a child region elected a `Head` which is a candidate for the
      // current region, we must take the same decision for parent region too.
      // If this is not done, we may end up disconnecting nodes from the entry,
      // because we do not have a single entry point into the tree of nested
      // regions. In other words, suppose that we elect for the outer `Region` A
      // as `Head`. If A is also contained in the inner child region, and we
      // elect another block, say B, as its `Head`, it would mean that A becomes
      // a late entry for the inner region, causing it to be disconnected (late
      // entry edges are transformed into `goto` edges). If these criteria dp
      // not apply, we continue with the standard criterion election.

      // All the blocks which have an incoming edge from a block not part of the
      // region itself, are considered as head candidates
      llvm::SmallMapVector<NodeT, size_t, 4> HeadCandidates;
      for (NodeT Block : getHeadCandidates(*CurrentRegion)) {
        HeadCandidates[Block]++;
      }

      // Criterion 1. Please note that we iterate just on the direct
      // `children` of each region, and not explicitly on all the
      // `grandchildren` too, but this is justified by the following reasoning:
      // Imagine we have the nested regions A B C (nesting is outer->inner), if
      // some node X is a late entry for C, either it is a late entry for B too
      // (and this in turns means that for being a late entry, it must have been
      // a candidate `Head` at some point, but not chosen as the elected
      // `Head`), or it is a late entry for C but not for B.
      // In the first scenario, since X is a late entry for B too, when we
      // analyze A, we will find it as a late entry for B, which is a direct
      // child of A, and so we are good. If, however, X is a late entry for C
      // but not for B, it means that we do not have an edge crossing multiple
      // layers from A to C (otherwise X would be a late entry for B too). In
      // that case X is contained in C, B, A by construction, but it is not a
      // late entry for B, therefore not a candidate `Head` for A. Therefore we
      // do not care to look at it when analyzing the candidate heads for A.
      for (auto ChildRegion : CurrentRegion->children()) {
        NodeT ChildHead = ChildRegion->getHead();
        revng_assert(ChildHead);

        for (NodeT Block : getHeadCandidates(*ChildRegion)) {
          if (Block != ChildHead) {
            HeadCandidates.erase(Block);
          }
        }
      }

      // Criterion 2. Please note that we just iterate over the direct
      // `children` of each region, and not explicitly on all the
      // `grandchildren` too, but this is justified by the following reasoning:
      // Suppose we have regions A B (nesting is outer->inner).
      // If we're looking at A, and one of A's candidates heads (X) is also in a
      // direct children region (B), do we need to look at B's children for
      // criterion 2?
      // The answer is no because if there is one of B's children that "forces"
      // B's head to X, it's enough to look at B. And if there isn't a children
      // of B that forces B's head to X, we don't care about it at all, because
      // we just need A to be coherent with B in this sense.
      for (auto ChildRegion : CurrentRegion->children()) {
        NodeT ChildHead = ChildRegion->getHead();
        revng_assert(ChildHead);

        // If one of our candidate heads is already the elected head of a child
        // region, we elect it as our head
        for (auto &[HeadCandidate, _] : HeadCandidates) {
          if (HeadCandidate == ChildHead) {
            CurrentRegion->setHead(HeadCandidate);
            break;
          }
        }
      }

      // If we elected an `Head` with the "shortcut" criterion mentioned above,
      // we skip the standard election phase
      if (CurrentRegion->getHead()) {
        continue;
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
              MaxNHead = NumIncoming;
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
template class GenericRegionInfo<Scope<llvm::Function *>>;
