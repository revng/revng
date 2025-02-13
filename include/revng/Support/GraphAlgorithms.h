#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <ranges>
#include <set>
#include <vector>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "revng/Support/Debug.h"

/// A generic way to compute a set of entry points to a graph such that any node
/// in said graph is reachable from at least one of those points.
template<typename GraphType>
  requires std::is_pointer_v<GraphType>
std::vector<typename llvm::GraphTraits<GraphType>::NodeRef>
entryPoints(GraphType &&Graph) {
  using NodeRef = typename llvm::GraphTraits<GraphType>::NodeRef;

  std::vector<NodeRef> Result;

  // First, find all SCCs reachable from nodes without predecessors
  std::set<const NodeRef> Visited;
  for (const auto &Node : llvm::nodes(Graph)) {
    const auto &Preds = llvm::children<llvm::Inverse<NodeRef>>(Node);
    // If the Node has predecessors, skip it for now. It will be reached by a
    // visit from its predecessor.
    if (Preds.begin() != Preds.end())
      continue;

    // Node has no predecessor, add it to Result.
    Result.push_back(Node);
    // Mark all the nodes reachable from it as Visited.
    for (const auto &Child : llvm::post_order_ext(NodeRef(Node), Visited))
      ;
  }

  // At this point, everything in Visited is reachable from the "easy" entry
  // points, e.g. the nodes without predecessors that we have just detected
  // above.
  for (auto *Node : llvm::nodes(Graph)) {
    if (Visited.contains(Node))
      continue;

    auto SCCBeg = llvm::scc_begin(NodeRef(Node));
    auto SCCEnd = llvm::scc_end(NodeRef(Node));
    // Ignore the case where there are no SCCs.
    if (SCCBeg == SCCEnd)
      continue;

    // Now we look only at the first SCC. We don't want to ever increment the
    // SCC iterator, because we want to only compute one SCC at a time, while
    // incrementing the SCC iterator computes the next SCC, possibly stepping
    // over stuff that has been Visited in the meantime.
    // For an example where this may happen, imagine the graph
    // A->B, B->A, B->C, C->D, D->C, where llvm::nodes visits D before A.
    // When visiting D, it would only see the SCC {C, D}, then when visiting A,
    // it would see the SCC {A, B} first, but it would recompute the SCC {C, D}
    // if incrementing the SCC iterator. This is something we want to avoid.
    const auto &TheSCC = *SCCBeg;
    const NodeRef &SCCEntryNode = TheSCC.front();

    // If the initial node of the SCC is Visited, it means that the whole SCC
    // was visited by one of the previous iterations, so we just ignore it.
    if (Visited.contains(SCCEntryNode))
      continue;

    // Then we mark all the nodes in the SCC as Visited, since we're visiting
    // them now.
    Visited.insert(TheSCC.begin(), TheSCC.end());

    // Now, let's try to figure out if this SCC is reachable from outside.
    // If it is NOT, then we have to add the first element of the SCC to
    // Results.
    bool HasPredecessorOutsideSCC = false;
    for (const NodeRef &SCCNode : TheSCC) {
      for (auto *PredNode : llvm::inverse_children<NodeRef>(SCCNode)) {
        if (llvm::find(TheSCC, PredNode) == TheSCC.end()) {
          HasPredecessorOutsideSCC = true;
          break;
        }
      }
      if (HasPredecessorOutsideSCC)
        break;
    }

    // If no element in TheSCC has a predecessor outside TheSCC we have to elect
    // an entry point for TheSCC. We just pick the first element since we have
    // no better clue about which entry would be best.
    if (not HasPredecessorOutsideSCC)
      Result.push_back(SCCEntryNode);
  }

  return Result;
}

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
auto graph_successors(GraphT Block) {
  return llvm::make_range(GT::child_begin(Block), GT::child_end(Block));
}

template<class GraphT>
size_t graphSuccessorsSize(GraphT Block) {
  auto Range = graph_successors(Block);
  return std::distance(Range.begin(), Range.end());
}

template<class GraphT>
auto graph_predecessors(GraphT Block) {
  using InverseGraphTraits = llvm::GraphTraits<llvm::Inverse<GraphT>>;
  return graph_successors<GraphT, InverseGraphTraits>(Block);
}

template<class GraphT>
auto graphPredecessorsSize(GraphT Block) {
  auto Range = graph_predecessors(Block);
  return std::distance(Range.begin(), Range.end());
}

namespace revng::detail {

/// `EdgeDescriptor` represents a pair of a `Source` and `Target` nodes
template<class NodeT>
using EdgeDescriptor = std::pair<NodeT, NodeT>;

enum FilterSet {
  WhiteList,
  BlackList
};

template<class NodeT>
class DFStack {
public:
  using InternalMapT = llvm::SmallMapVector<NodeT, bool, 4>;

private:
  InternalMapT InternalMap;

public:
  // Return the insertion iterator over the underlying map
  std::pair<typename InternalMapT::iterator, bool> insertInMap(NodeT Block,
                                                               bool OnStack) {
    return InternalMap.insert(std::make_pair(Block, OnStack));
  }

  // Return true if Block is currently on the active stack of visit
  bool onStack(NodeT Block) const {
    auto Iter = InternalMap.find(Block);
    return Iter != InternalMap.end() && Iter->second;
  }

  // Invoked after we have processed all children of a node during the DFS
  void completed(NodeT Block) { InternalMap[Block] = false; }

  InternalMapT::iterator begin() { return InternalMap.begin(); }

  InternalMapT::iterator end() { return InternalMap.end(); }
};

template<FilterSet FilterSetType, class NodeT>
class DFStackBackedge {
  using InternalMapT = DFStack<NodeT>::InternalMapT;

private:
  DFStack<NodeT> VisitStack;
  llvm::SmallPtrSet<NodeT, 4> &Set;

public:
  DFStackBackedge(llvm::SmallPtrSet<NodeT, 4> &Set) : Set(Set) {}

  std::pair<typename InternalMapT::iterator, bool> insert(NodeT Block) {

    // If we are trying to insert a valid block, we can process it as normal
    if ((FilterSetType == FilterSet::WhiteList and Set.contains(Block))
        or (FilterSetType == FilterSet::BlackList
            and not Set.contains(Block))) {
      return VisitStack.insertInMap(Block, true);
    } else {

      // We want to completely ignore the fact that we inserted an element in
      // the `DFStack`, otherwise we will explore it anyway, therefore we
      // manually return `false`, so the node is not explored at all, in
      // addition of adding it as not on the exploration stack
      auto InsertIt = VisitStack.insertInMap(Block, false);
      return std::make_pair(InsertIt.first, false);
    }
  }

  bool onStack(NodeT Block) const { return VisitStack.onStack(Block); }

  void completed(NodeT Block) { return VisitStack.completed(Block); }
};

} // namespace revng::detail

// TODO: remove the `BlackList`/`WhiteList` parameter once the last use is
//       dropped, checking that no other uses are still present
template<revng::detail::FilterSet FilterSetType,
         class GraphT,
         class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSetVector<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedgesImpl(GraphT Block,
                 llvm::SmallPtrSet<typename GT::NodeRef, 4> &Set) {
  using NodeRef = typename GT::NodeRef;
  using StateType = typename revng::detail::DFStackBackedge<FilterSetType,
                                                            NodeRef>;
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeRef>;
  StateType State(Set);

  llvm::SmallSetVector<EdgeDescriptor, 4> Backedges;

  // Declare manually a custom `df_iterator`
  using bdf_iterator = llvm::df_iterator<GraphT, StateType, true, GT>;
  auto Begin = bdf_iterator::begin(Block, State);
  auto End = bdf_iterator::end(Block, State);

  for (NodeRef Block : llvm::make_range(Begin, End)) {
    for (NodeRef Succ :
         llvm::make_range(GT::child_begin(Block), GT::child_end(Block))) {
      if (State.onStack(Succ)) {
        Backedges.insert(EdgeDescriptor(Block, Succ));
      }
    }
  }

  return Backedges;
}

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSetVector<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedges(GraphT Block) {
  llvm::SmallPtrSet<typename GT::NodeRef, 4> EmptySet;
  return getBackedgesImpl<revng::detail::FilterSet::BlackList>(Block, EmptySet);
}

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSetVector<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedgesWhiteList(GraphT Block,
                      llvm::SmallPtrSet<typename GT::NodeRef, 4> &Set) {
  return getBackedgesImpl<revng::detail::FilterSet::WhiteList>(Block, Set);
}

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSetVector<revng::detail::EdgeDescriptor<typename GT::NodeRef>, 4>
getBackedgesBlackList(GraphT Block,
                      llvm::SmallPtrSet<typename GT::NodeRef, 4> &Set) {
  return getBackedgesImpl<revng::detail::FilterSet::BlackList>(Block, Set);
}

template<class GraphT, class GT>
bool hasSuccessor(GraphT Pred, GraphT Succ) {
  for (auto Child :
       llvm::make_range(GT::child_begin(Pred), GT::child_end(Pred))) {
    if (Child == Succ) {
      return true;
    }
  }

  return false;
}

/// The `findReachableNodes` primitive returns the set of nodes reachable from
/// the `Source` node, stopping at the `Stop` node.
/// \param Source The node where the DFS starts from
/// \param Stop The node where the DFS should stop, if present
/// \return Note that the return value is a `SmallSetVector`, so we have
///         additional guarantees about the deterministic iteration order over
///         the returned set
template<class GraphT, class GT>
llvm::SmallSetVector<typename llvm::GraphTraits<GraphT>::NodeRef, 4>
findReachableNodesImpl(GraphT Source, GraphT Stop = nullptr) {
  using NodeRef = typename GT::NodeRef;

  using SmallSetVector = llvm::SmallSetVector<NodeRef, 4>;
  SmallSetVector DFSNodes;

  // Verify that the `Source` node is valid
  revng_assert(Source != nullptr);

  // Perform the forward DFS visit, stopping at the `Stop` node, if present
  using ExtType = typename llvm::df_iterator_default_set<NodeRef>;
  ExtType ExtSet;
  if (Stop != nullptr) {
    ExtSet.insert(Stop);
  }

  using df_iterator = llvm::df_iterator<GraphT, ExtType, false, GT>;
  auto Begin = df_iterator::begin(Source, ExtSet);
  auto End = df_iterator::end(Source, ExtSet);

  for (NodeRef Block : llvm::make_range(Begin, End)) {
    DFSNodes.insert(Block);
  }

  return DFSNodes;
}

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
inline llvm::SmallSetVector<typename llvm::GraphTraits<GraphT>::NodeRef, 4>
findReachableNodes(GraphT Source, GraphT Stop = nullptr) {
  return findReachableNodesImpl<GraphT, GT>(Source, Stop);
}

/// The `nodesBetween` primitive, collects all the nodes present on any path
/// that connects the `Source` node and the `Target` node.
/// \return Note that the return value is a `SmallSetVector`, so we have
///         additional guarantees about the deterministic iteration order over
///         the returned set
template<class GraphT, class GraphDirection>
llvm::SmallSetVector<typename llvm::GraphTraits<GraphT>::NodeRef, 4>
nodesBetweenImpl(GraphT Source, GraphT Target) {
  using FGT = llvm::GraphTraits<GraphDirection>;
  using BGT = llvm::GraphTraits<llvm::Inverse<GraphDirection>>;
  using NodeRef = typename FGT::NodeRef;
  using SmallSetVector = llvm::SmallSetVector<NodeRef, 4>;

  // We cannot perform a `nodesBetween` search either starting from a null
  // `Source` or ending in a null `Target`
  revng_assert(Source != nullptr);
  revng_assert(Target != nullptr);

  // Perform the forward DFS visit, stopping at the `Target` node
  SmallSetVector ForwardDFSNodes = findReachableNodes<NodeRef, FGT>(Source,
                                                                    Target);

  // Perform the backward DFS visit, stopping at the `Source` node
  SmallSetVector BackwardDFSNodes = findReachableNodes<NodeRef, BGT>(Target,
                                                                     Source);

  // Perform the set intersection between the forward and backward set nodes
  SmallSetVector Result = llvm::set_intersection(ForwardDFSNodes,
                                                 BackwardDFSNodes);

  // Corner case handling
  if (not Result.empty()) {

    // We found some nodes in the intersection of the two DFSs, therefore we
    // actually have some reachables nodes, and we add both the `Source` and the
    // `Target` in the `Result` set (they are not automatically included because
    // the are part of the respective ext sets, and therefore not present in the
    // intersection)
    Result.insert(Target);
    Result.insert(Source);
  } else if (hasSuccessor<GraphT, FGT>(Source, Target)) {

    // If the `Target` is a direct successor of `Source`, we may have that the
    // resulting intersection is empty, if the only path connecting them is the
    // trivial one, but the reachable nodes are actually `Source` and `Target`
    revng_assert(Result.empty());
    Result.insert(Target);
    Result.insert(Source);
  } else if (Source == Target) {

    // If `Source` is equal to `Target`, due to fact that `Target` (which is `==
    // Source`) is added to the ext set, (and vice-versa), we will not find any
    // candidate note in our exploration, therefore we need to manually add it
    // to the `Result` set
    Result.insert(Source);
  }

  // The only situation where we can have a `Result` set of dimension 1, is when
  // `Source` == `Target`.
  revng_assert(Source == Target or Result.size() != 1);

  return Result;
}

template<class GraphT>
inline llvm::SmallSetVector<typename llvm::GraphTraits<GraphT>::NodeRef, 4>
nodesBetween(GraphT Source, GraphT Destination) {
  return nodesBetweenImpl<GraphT, GraphT>(Source, Destination);
}

template<class GraphT>
inline llvm::SmallSetVector<typename llvm::GraphTraits<GraphT>::NodeRef, 4>
nodesBetweenReverse(GraphT Source, GraphT Destination) {
  using namespace llvm;
  return nodesBetweenImpl<GraphT, Inverse<GraphT>>(Source, Destination);
}

template<class GraphT>
bool isDAG(GraphT Graph) {
  for (llvm::scc_iterator<GraphT> I = llvm::scc_begin(Graph),
                                  IE = llvm::scc_end(Graph);
       I != IE;
       ++I) {
    if (I.hasCycle())
      return false;
  }
  return true;
}
