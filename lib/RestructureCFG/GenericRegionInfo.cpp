//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>

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
static Logger Log("generic-region-info");

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

struct HeadScoreInfo {
  size_t NumEdgesFromAncestors = 0;
  size_t NumEdgesFromDirectParent = 0;
  size_t NumEdgesFromSelf = 0;
  size_t NumEdgesFromChildren = 0;
  bool IsInChild = false;
  bool IsChildHead = false;
};

static void logScoreInfo(const HeadScoreInfo &Info) {
  if (Log.isEnabled()) {
    LoggerIndent Indent{ Log };
    revng_log(Log, "NumEdgesFromAncestors:    " << Info.NumEdgesFromAncestors);
    revng_log(Log,
              "NumEdgesFromDirectParent: " << Info.NumEdgesFromDirectParent);
    revng_log(Log, "IsInChild:                " << Info.IsInChild);
    revng_log(Log, "IsChildHead:              " << Info.IsChildHead);
    revng_log(Log, "NumEdgesFromSelf:         " << Info.NumEdgesFromSelf);
    revng_log(Log, "NumEdgesFromChildren:     " << Info.NumEdgesFromChildren);
  }
}

/// Helper static function which computes the `Head` candidates for a given
/// region
template<class NodeT>
static llvm::SmallMapVector<NodeT, HeadScoreInfo, 4>
getHeadCandidatesInfo(GenericRegion<NodeT> &Region) {
  llvm::SmallMapVector<NodeT, HeadScoreInfo, 4> HeadCandidatesInfo;
  GenericRegion<NodeT> *Parent = Region.getParent();
  for (NodeT Block : Region.blocks()) {
    for (NodeT Predecessor : graph_predecessors(Block)) {
      // If the Region does not contain the Predecessor, and the Predecessor is
      // strictly in the Parent region, the Block is a Candidate.
      // Predecessors not strictly in the parent region but just in an ancestor
      // don't count to make the Block a candidate.
      // The reason why they don't is that if we pick a head that has no
      // predecessors in in the parent region, all edges from parent region to
      // this region will become late entris, hence gotos.
      // So, if the head of the parent region is then elected to be a node that
      // is *not* in the current child region, it will not be possible to reach
      // the child from the elected head of the parent, except via gotos. Hence,
      // that would disconnect the child from the parent, which is something we
      // want to avoid by design.
      if (not Region.containsBlock(Predecessor)) {
        if (not Parent or Parent->containsBlock(Predecessor)) {
          HeadCandidatesInfo[Block].NumEdgesFromDirectParent++;
        }
      }
    }
  }
  revng_assert(not HeadCandidatesInfo.empty());

  for (auto &[Node, Info] : HeadCandidatesInfo) {
    for (auto *ChildRegion : Region.children()) {
      if (ChildRegion->containsBlock(Node)) {
        Info.IsInChild = true;
        for (auto &GrandChild : post_order(ChildRegion)) {
          if (ChildRegion->getHead() == Node)
            Info.IsChildHead = true;
        }
      }
    }
    for (NodeT Predecessor : graph_predecessors(Node)) {
      bool PredecessorInChild = false;

      if (GenericRegion<NodeT> *Parent = Region.getParent();
          Parent and not Parent->containsBlock(Predecessor)) {
        Info.NumEdgesFromAncestors++;
      }

      for (auto *ChildRegion : Region.children()) {
        if (ChildRegion->containsBlock(Predecessor)) {
          Info.NumEdgesFromChildren++;
          PredecessorInChild = true;
        }
      }
      if (not PredecessorInChild) {
        if (Region.containsBlock(Predecessor)) {
          Info.NumEdgesFromSelf++;
        }
      }
    }
  }

  return HeadCandidatesInfo;
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

/// Stateful visitor used to explore a `GenericRegion` starting from a
/// candidate head.
///
/// The visit never leaves the region, and it never traverses edges go to a Node
/// inside a child region. If it reaches the child regions in its elected head,
/// the head is valid also for the parent. If it reaches the child region in a
/// node that is different from its elected head, the head is not valid.
static HeadScoreInfo worst() {
  return HeadScoreInfo{
    .NumEdgesFromAncestors = 0,
    .NumEdgesFromDirectParent = 0,
    .NumEdgesFromSelf = 0,
    .NumEdgesFromChildren = 0,
    .IsInChild = true,
    .IsChildHead = true,
  };
}

static std::strong_ordering isLess(const HeadScoreInfo &CurrentBest,
                                   const HeadScoreInfo &Candidate) {

  // We want to pick a head that, in a way that has the best chances of
  // reducing the number of gotos we emit when dagifying.
  // DAGify emits gotos for late entries and for retreating edges.
  // At this stage, we can't reason about retreating edges that point to a
  // node that is different from head, because that would require
  // effectively removing edges and performing new visits.
  //
  // In general, gotos on retreating edges are considered worse than gotos
  // on forward edges. So we use first criteria that reduce the number of
  // gotos on retreating edges and then criteria that reduce the number of
  // gotos on forward edges.
  // Also, among retreating edges, turning a goto into a continue_to is
  // considered worse than turning it into a proper continue.
  //
  // We consider the following criteria, in this order
  // 1. We try to maximize the number of incoming edges to the head from
  //    within the region or one of its children. These are all rendered as
  //    `continue` or `continue_to` and are all strictly better than just
  //    `goto`.
  auto CurrentInnerBackedges = CurrentBest.NumEdgesFromSelf
                               + CurrentBest.NumEdgesFromChildren;
  auto CandidateInnerBackedges = Candidate.NumEdgesFromSelf
                                 + Candidate.NumEdgesFromChildren;
  if (auto Cmp = CurrentInnerBackedges <=> CandidateInnerBackedges; Cmp != 0) {
    return Cmp;
  }

  // 2. We try to maximize the number of incoming edges to the head from
  //    strictly within the region. These are all rendered as `continue`
  //    and are all strictly better than just `goto`.
  //    The reason why we consider this only after 1. is that we consider
  //    less forward `goto`s always better, even if some are `continue_to`
  //    and not proper `continue`.
  if (auto Cmp = CurrentBest.NumEdgesFromSelf <=> Candidate.NumEdgesFromSelf;
      Cmp != 0) {
    return Cmp;
  }

  // 3. We try to maximize the number of incoming edges to the head from its
  //    parent region or an ancestor region. These will turn a forward
  //    `goto` for a late entry in a regular entry in the region.
  auto CurrentAncestorEntries = CurrentBest.NumEdgesFromAncestors
                                + CurrentBest.NumEdgesFromDirectParent;
  auto CandidateAncestorEntries = Candidate.NumEdgesFromAncestors
                                  + Candidate.NumEdgesFromDirectParent;
  if (auto Cmp = CurrentAncestorEntries <=> CandidateAncestorEntries;
      Cmp != 0) {
    return Cmp;
  }

  // 4. We try to maximize the number of incoming edges to the head from its
  //    strict parent region. These will turn a forward `goto` for a late
  //    entry in a regular entry in the region.
  auto CurrentFromDirect = CurrentBest.NumEdgesFromDirectParent;
  auto CandidateFromDirect = Candidate.NumEdgesFromDirectParent;
  if (auto Cmp = CurrentFromDirect <=> CandidateFromDirect; Cmp != 0) {
    return Cmp;
  }

  // 5. We favor heads that are not nested in children.
  //    If we pick a head that is nested in a children, all the backedges
  //    inside the children that point to that head will become
  //    `continue_to`, which is strictly worse than a `continue`. This would
  //    subvert a choice of a child region, and we try to avoid it if
  //    possible because the consequences are not clear.
  if (auto Cmp = CurrentBest.IsInChild <=> Candidate.IsInChild; Cmp != 0) {
    if (Cmp == std::strong_ordering::less)
      return std::strong_ordering::greater;
    else
      return std::strong_ordering::less;
  }

  revng_assert(CurrentBest.IsChildHead == Candidate.IsChildHead);

  // TODO: in case all head candidates are inside a child region we didn't
  // put much effort in improving this heuristic. See the TODO in criterion
  // 5, which is the only one considering head candidates inside a child.
  return std::strong_ordering::equivalent;
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

      if (Log.isEnabled()) {
        revng_log(Log, "Blocks:");
        for (NodeT Block : CurrentRegion->blocks()) {
          LoggerIndent BlockIndent{ Log };
          revng_log(Log, Block->getName());
        }
      }

      // The `Head` election phase works in a bottom-up fashion and it must
      // guarantee that the decision we take when processing a region, is
      // coherent with all the children region it contains. Specifically, we
      // must be coherent in terms of _late entries_. This means that if a
      // node is considered a late entry for a child region, it must be a late
      // entry for its parent region as well. So it may not be elected as a head
      // for the parent. If a late entry for a child is elected as head for the
      // parent we may end up disconnecting portion of the graph from the entry.
      // In practice, this means that when electing the `Head` of a region, we
      // must exclude from the candidates all the nodes that happens to be late
      // entries for its children regions.
      // Once we have guaranteed this, we can pick whatever candidate head is
      // left with a logic of our choice.

      // All the blocks which have an incoming edge from the direct parent
      // region are considered head candidates.
      auto HeadCandidatesInfo = getHeadCandidatesInfo(*CurrentRegion);
      revng_log(Log, "Head candidates info:");
      for (auto &[Node, Info] : HeadCandidatesInfo) {
        LoggerIndent CandidateInfoIndent{ Log };
        revng_log(Log, Node->getName());
        logScoreInfo(Info);
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
      {
        LoggerIndent PurgeIndent{ Log };

        SmallVector<NodeT> CandidatesToPurge;
        for (auto &[Node, Info] : HeadCandidatesInfo) {
          if (Info.IsInChild and not Info.IsChildHead) {
            CandidatesToPurge.push_back(Node);
          }
        }
        for (NodeT ToPurge : CandidatesToPurge) {
          LoggerIndent ToPurgeIndent{ Log };
          revng_log(Log,
                    "child's late entry block can't be head of parent: "
                      << ToPurge->getName());
          HeadCandidatesInfo.erase(ToPurge);
        }
        revng_assert(not HeadCandidatesInfo.empty());
        if (not CandidatesToPurge.empty()) {
          revng_log(Log, "Remaining Head candidates:");
          for (const auto &[Block, _] : HeadCandidatesInfo) {
            LoggerIndent CandidateIndent{ Log };
            revng_log(Log, Block->getName());
          }
        }
      }

      revng_log(Log, "Pick the best head");
      NodeT CurrentHead = nullptr;
      HeadScoreInfo Best = worst();

      size_t MaxIncomings = 0;
      for (NodeT Block : RPOT) {
        LoggerIndent IndentRPOT{ Log };

        auto HeadIt = HeadCandidatesInfo.find(Block);
        if (HeadIt == HeadCandidatesInfo.end())
          continue;

        auto &[Candidate, CandidateScore] = *HeadIt;
        auto Cmp = isLess(Best, CandidateScore);
        // If Best is still larger than CandidateScore, go to next
        if (Cmp > 0)
          continue;

        if (Cmp < 0) {
          if (isValidHead(*CurrentRegion, Candidate)) {
            revng_log(Log, "New Head: " << Candidate->getName());
            revng_log(Log, "New Best:");
            logScoreInfo(CandidateScore);
            Best = CandidateScore;
            CurrentHead = Candidate;
          } else {
            revng_log(Log, "Invalid head: " << Candidate->getName());
          }
        } else {
          // 6. As a fallback, we pick the node with the shortest path from
          // entry.
          //
          // Compute the `ShortestPathFromEntry` map since we need to
          // break a tie here
          if (not ShortestPathFromEntry.has_value()) {
            ShortestPathFromEntry = computeShortesPath(F);
          }
          size_t CurrentShortest = mapAt(*ShortestPathFromEntry, CurrentHead);
          size_t CandidateShortest = mapAt(*ShortestPathFromEntry, Candidate);
          if (CandidateShortest < CurrentShortest) {
            if (isValidHead(*CurrentRegion, Candidate)) {
              revng_log(Log,
                        "New Head with shortest path from entry: "
                          << Candidate->getName() << ": " << CandidateShortest);

              CurrentHead = Candidate;
            } else {
              revng_log(Log, "Invalid head: " << Candidate->getName());
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
