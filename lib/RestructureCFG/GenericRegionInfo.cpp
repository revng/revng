//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GenericCycleImpl.h"
#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
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
Logger Log("generic-region-info");

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

/// Stateful visitor used to explore a `GenericRegion` starting from a
/// candidate head.
///
/// The visit never leaves the region, and it never traverses edges go to a Node
/// inside a child region. If it reaches the child regions in its elected head,
/// the head is valid also for the parent. If it reaches the child region in a
/// node that is different from its elected head, the head is not valid.
template<class GraphT, class GT>
class RegionVisitor {
public:
  using NodeT = typename GT::NodeRef;
  using Region = GenericRegion<NodeT>;

private:
  Region &TheRegion;
  llvm::SmallPtrSet<NodeT, 8> Visited;
  llvm::SmallVector<NodeT, 8> WorkList;

public:
  RegionVisitor(Region &TheRegion) : TheRegion(TheRegion) {}

public:
  bool isValidHead(NodeT HeadCandidate) {
    Visited.clear();
    Visited.insert(HeadCandidate);
    WorkList.clear();
    WorkList.push_back(HeadCandidate);

    while (not WorkList.empty()) {
      NodeT Current = WorkList.pop_back_val();
      for (NodeT Successor : llvm::children<GraphT>(Current)) {

        // We never visit anything outside the region
        if (not TheRegion.containsBlock(Successor))
          continue;

        // If we find an edge towards a children region, we don't traverse it
        // unless it goes to the elected head.
        if (isLateEntryOfChild(Current, Successor))
          continue;

        if (Visited.insert(Successor).second)
          WorkList.push_back(Successor);
      }
    }
    return Visited.size() == TheRegion.size();
  }

private:
  bool isLateEntryOfChild(NodeT Source, NodeT Target) const {
    for (Region *Child : TheRegion.children()) {
      revng_assert(nullptr != Child->getHead());
      if (Child->containsBlock(Target) and Target != Child->getHead()
          and not Child->containsBlock(Source)) {
        return true;
      }
    }
    return false;
  }
};

template<class GraphT, class GT>
bool GenericRegionInfo<GraphT, GT>::isValidHead(Region &CurrentRegion,
                                                NodeT Candidate) {
  return RegionVisitor<GraphT, GT>(CurrentRegion).isValidHead(Candidate);
}

template<class GraphT, class GT>
void GenericRegionInfo<GraphT, GT>::electHead(GraphT F) {

  llvm::SmallVector<NodeT> RPOT;
  llvm::copy(llvm::post_order(F), std::back_inserter(RPOT));

  // We use the shortest distance from the entry block of the function as a tie
  // breaker. Therefore, we delay its computation until it is necessary.
  std::optional<llvm::SmallDenseMap<NodeT, size_t>>
    ShortestPathFromEntry = std::nullopt;

  // Perform the head election for each `Region`, in a bottom up fashion
  size_t RegionIndex = 0;
  for (auto &TopLevelRegion : top_level_regions()) {
    LoggerIndent IndentRegion{ Log };
    for (auto &CurrentRegion : post_order(&TopLevelRegion)) {
      revng_log(Log,
                "DAGify processing region with index: "
                  << std::to_string(RegionIndex++) << "\n");
      LoggerIndent MoreIndentRegion{ Log };

      // The `Head` election phase works in a bottom-up fashion and it must
      // guarantee that the decision we take when processing a region, is
      // coherent with all the children region it contains. Specifically, we
      // must guarantee the following properties.
      // * We must be coherent in terms of _late entries_. This means that if a
      // node is considered a late entry for a child region, it must be a late
      // entry for its parent region as well. So it may not be elected as a head
      // for the parent. If a late entry for a child is elected as head for the
      // parent we may end up disconnecting portion of the graph from the entry.
      // In practice, this means that when electing the `Head` of a region, we
      // must exclude from the candidates all the nodes that happens to be late
      // entries for its children regions.
      // * If a child region elected a `Head` is also a candidate head for the
      // current region, we must take the same decision for parent region too.
      // If this is not done, we may end up disconnecting nodes from the entry,
      // because we do not have a single entry point into the tree of nested
      // regions. In other words, suppose that we elect for the outer `Region` A
      // as `Head`. If A is also contained in the inner child region, and we
      // elect another block, say B, as its `Head`, it would mean that A becomes
      // a late entry for the inner region, causing it to be disconnected (late
      // entry edges are transformed into `goto` edges).
      //
      // Once we have guaranteed these properties, we can pick whatever
      // candidate head is left with a logic of our choice.

      // All the blocks which have an incoming edge from a block not part of the
      // region itself, are considered as head candidates
      revng_log(Log, "Head candidates:");
      llvm::SmallMapVector<NodeT, size_t, 4> HeadCandidates;
      for (NodeT Block : getHeadCandidates(*CurrentRegion)) {
        LoggerIndent CandidateIndent{ Log };
        revng_log(Log, Block->getName());
        HeadCandidates[Block]++;
      }

      // Filter away children's late entries. If there are any children, their
      // head has already been elected. None of the nodes in a children
      // different from the children's already selected head can be selected as
      // a head of the parent, because that would cause a regular entry in the
      // parent to also be a late entry in the children, which is impossible.
      // Notice that we iterate only on direct children regions and not on
      // grandchildren but given that we work from the innermost to the
      // outermost regions the property is guaranteed by induction.
      revng_log(Log,
                "Purging childrens' late entries from parent's candidates");
      for (auto ChildRegion : CurrentRegion->children()) {
        LoggerIndent ChildrenIndent{ Log };
        NodeT ChildHead = ChildRegion->getHead();
        revng_assert(ChildHead);

        for (NodeT Block : getHeadCandidates(*ChildRegion)) {
          if (Block != ChildHead) {
            HeadCandidates.erase(Block);
            revng_log(Log,
                      "child's late entry block can't be head of parent: "
                        << Block->getName());
          }
        }
        revng_log(Log, "Remaining Head candidates:");
        for (const auto &[Block, _] : HeadCandidates) {
          LoggerIndent CandidateIndent{ Log };
          revng_log(Log, Block->getName());
        }
      }

      // If one of the HeadCandidates is already an elected head of a child
      // region, pick it as a head for the parent region as well.
      // Notice that again we iterate only on direct children regions and not on
      // grandchildren but given that we work from the innermost to the
      // outermost regions the property is guaranteed by induction.
      revng_log(Log, "Pick head that is already a head of a child");
      for (auto ChildRegion : CurrentRegion->children()) {
        LoggerIndent ChildrenIndent{ Log };
        NodeT ChildHead = ChildRegion->getHead();
        revng_assert(ChildHead);

        // If one of our candidate heads is already the elected head of a child
        // region, we elect it as our head
        for (auto &[HeadCandidate, _] : HeadCandidates) {
          if (HeadCandidate == ChildHead) {
            CurrentRegion->setHead(HeadCandidate);
            revng_log(Log, "setHead: " << HeadCandidate->getName());
            break;
          }
        }
      }

      // If we elected a `Head` of a child, we can move on to the next region.
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
      // A candidate that would not reach the whole region once the children
      // regions have been dagified cannot be elected. We only check this for
      // the candidate we are about to pick, and, if it turns out to be
      // invalid, we drop it and pick again.
      revng_log(Log, "Pick the best head");
      NodeT CurrentHead = nullptr;
      size_t MaxIncomings = 0;
      for (NodeT Block : RPOT) {
        LoggerIndent IndentRPOT{ Log };

        auto HeadIt = HeadCandidates.find(Block);
        if (HeadIt == HeadCandidates.end())
          continue;

        const auto &[HeadCandidate, NumIncoming] = *HeadIt;
        if (NumIncoming < MaxIncomings)
          continue;

        if (NumIncoming > MaxIncomings) {
          if (isValidHead(*CurrentRegion, HeadCandidate)) {
            MaxIncomings = NumIncoming;
            CurrentHead = HeadCandidate;
            revng_log(Log, "New Max Incomings: " << MaxIncomings);
            revng_log(Log, "New Head: " << CurrentHead->getName());
          }
        } else if (NumIncoming == MaxIncomings) {
          // Compute the `ShortestPathFromEntry` map since we need to
          // break a tie here
          if (not ShortestPathFromEntry.has_value()) {
            ShortestPathFromEntry = computeShortesPath(F);
          }
          size_t CurrentShortest = mapAt(*ShortestPathFromEntry, CurrentHead);
          size_t CandidateShortest = mapAt(*ShortestPathFromEntry,
                                           HeadCandidate);
          if (CandidateShortest < CurrentShortest) {
            if (isValidHead(*CurrentRegion, HeadCandidate)) {
              CurrentHead = HeadCandidate;
              revng_log(Log,
                        "New Head coming first in RPOT: "
                          << HeadCandidate->getName());
            }
          }
        }
      }
      // Verify that we found a Head
      revng_assert(CurrentHead != nullptr);

      // Set the Head for the `Region`
      CurrentRegion->setHead(CurrentHead);
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
  revng_log(Log, print());
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
