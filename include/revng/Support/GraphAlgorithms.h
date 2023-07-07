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

template<typename T>
struct scc_iterator_traits {
  using iterator = llvm::scc_iterator<T, llvm::GraphTraits<T>>;
  using iterator_category = std::forward_iterator_tag;
  using reference = decltype(*llvm::scc_begin((T) nullptr));
  using value_type = std::remove_reference_t<reference>;
  using pointer = value_type *;
  using difference_type = size_t;
};

template<typename NodeTy>
auto exitless_scc_range(NodeTy Entry) {
  using namespace llvm;

  auto Range = make_range(scc_begin(Entry), scc_end(Entry));

  using NodesVector = std::vector<NodeTy>;
  using GT = llvm::GraphTraits<NodeTy>;

  auto Filter = [](const NodesVector &SCC) {
    std::set<NodeTy> SCCNodes;
    SCCNodes.clear();
    for (NodeTy BB : SCC)
      SCCNodes.insert(BB);

    bool HasExit = false;
    bool AtLeastOneEdge = false;
    for (NodeTy BB : SCC) {
      auto Successors = make_range(GT::child_begin(BB), GT::child_end(BB));
      for (NodeTy Successor : Successors) {
        AtLeastOneEdge = true;
        if (!SCCNodes.contains(Successor)) {
          HasExit = true;
          break;
        }
      }

      if (HasExit)
        break;
    }

    return (not HasExit) and AtLeastOneEdge;
  };

  return make_filter_range(Range, Filter);
}

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

  // Return true if b is currently on the active stack of visit
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

template<FilterSet FilterSetType, class NodeT>
class DFStackReachable {
  using InternalMapT = DFStack<NodeT>::InternalMapT;

  template<typename T, unsigned S = 4>
  using SmallSetVector = llvm::SmallSetVector<T, S>;

private:
  DFStack<NodeT> VisitStack;

private:
  // Set which contains the desired targets nodes marked as reachable during
  // the visit
  SmallSetVector<NodeT> Targets;
  llvm::SmallPtrSet<NodeT, 4> &Set;
  llvm::SmallDenseMap<NodeT, SmallSetVector<NodeT>, 4> AdditionalNodes;
  NodeT Source = nullptr;
  NodeT Target = nullptr;
  bool FirstInvocation = true;

public:
  DFStackReachable(llvm::SmallPtrSet<NodeT, 4> &Set) : Set(Set) {}

  // Insert the initial target node at the beginning of the visit
  void insertTarget(NodeT Block) { Targets.insert(Block); }

  // Assign the `Source` node
  void assignSource(NodeT Block) { Source = Block; }

  // Assign the `Target` node
  void assignTarget(NodeT Block) { Target = Block; }

  SmallSetVector<NodeT> getReachables() { return Targets; }

  SmallSetVector<NodeT> &getAdditional(NodeT Block) {
    return AdditionalNodes[Block];
  }

  // Customize the `insert` method, in order to add the reachable nodes during
  // the DFS
  std::pair<typename InternalMapT::iterator, bool> insert(NodeT Block) {

    if ((FilterSetType == FilterSet::WhiteList and Set.contains(Block))
        or (FilterSetType == FilterSet::BlackList
            and not Set.contains(Block))) {

      // We need to insert the `Source` node, which is the first element on
      // which the `insert` method is called, only once, and later on skip it,
      // otherwise we may loop back from the `Source` and add additional nodes
      revng_assert(Source != nullptr);
      if (!FirstInvocation and Block == Source) {
        return VisitStack.insertInMap(Block, false);
      }
      FirstInvocation = false;

      // Check that, if we are trying to insert a block which is the `Targets`
      // set, we add all the nodes on the current visiting stack in the
      // `Targets` set
      auto SuccessorRange = graph_successors(Block);
      if (Targets.contains(Block)
          or (Target == nullptr and std::ranges::empty(SuccessorRange))) {
        for (auto const &[K, V] : VisitStack) {
          if (V) {
            Targets.insert(K);
          }
        }
      }

      // When we encounter a loop, we add to the additional set of nodes the
      // nodes that are onStack, for later additional post-processing
      if (VisitStack.onStack(Block)) {
        SmallSetVector<NodeT> &AdditionalSet = AdditionalNodes[Block];
        for (auto const &[K, V] : VisitStack) {
          if (V) {
            AdditionalSet.insert(K);
          }
        }
      }

      // Return the insertion iterator as usual
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

  std::pair<typename InternalMapT::iterator, bool> insertInMap(NodeT Block,
                                                               bool OnStack) {
    return VisitStack.insertInMap(Block, OnStack);
  }

  void completed(NodeT Block) { return VisitStack.completed(Block); }
};

} // namespace revng::detail

// TODO: remove the `BlackList`/`WhiteList` parameter once the last use in
//       `revng-c` is dropped, checking that no other uses are still present
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

// TODO: remove the `BlackList`/`WhiteList` parameter once the new AVI
//       implementation is merged, checking that no other uses are still present
template<revng::detail::FilterSet FilterSetType,
         class GraphT,
         class GT = llvm::GraphTraits<GraphT>>
llvm::SmallSetVector<typename GT::NodeRef, 4>
nodesBetweenImpl(GraphT Source,
                 GraphT Target,
                 llvm::SmallPtrSet<typename GT::NodeRef, 4> &Set) {
  using NodeRef = typename GT::NodeRef;
  using StateType = typename revng::detail::DFStackReachable<FilterSetType,
                                                             NodeRef>;
  StateType State(Set);

  using SmallSetVector = llvm::SmallSetVector<NodeRef, 4>;

  // Assign the `Source` node
  State.assignSource(Source);

  // Initialize the visited set with the target node, which is the boundary
  // that we don't want to trepass when finding reachable nodes
  State.assignTarget(Target);
  State.insertTarget(Target);
  State.insertInMap(Target, false);

  using nbdf_iterator = llvm::df_iterator<GraphT, StateType, true, GT>;
  auto Begin = nbdf_iterator::begin(Source, State);
  auto End = nbdf_iterator::end(Source, State);

  for (NodeRef Block : llvm::make_range(Begin, End)) {
    (void) Block;
  }

  auto Targets = State.getReachables();
  // Add in a fixed point fashion the additional nodes
  SmallSetVector OldTargets;
  do {
    // At each iteration obtain a copy of the old set, so that we are able to
    // exit from the loop as soon no change is made to the `Targets` set

    OldTargets = Targets;

    // Temporary storage for the nodes to add at each iteration, to avoid
    // invalidation on the `Targets` set
    SmallSetVector NodesToAdd;

    for (NodeRef Block : Targets) {
      SmallSetVector &AdditionalSet = State.getAdditional(Block);
      NodesToAdd.insert(AdditionalSet.begin(), AdditionalSet.end());
    }

    // Add all the additional nodes found in this step
    Targets.insert(NodesToAdd.begin(), NodesToAdd.end());
    NodesToAdd.clear();

  } while (Targets != OldTargets);

  return Targets;
}

template<class GraphT>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetween(GraphT Source, GraphT Destination) {
  llvm::SmallPtrSet<typename llvm::GraphTraits<GraphT>::NodeRef, 4> EmptySet;
  return nodesBetweenImpl<revng::detail::FilterSet::BlackList,
                          GraphT,
                          llvm::GraphTraits<GraphT>>(Source,
                                                     Destination,
                                                     EmptySet);
}

template<class GraphT,
         typename NodeRef = typename llvm::GraphTraits<GraphT>::NodeRef>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetweenWhiteList(GraphT Source,
                      GraphT Destination,
                      llvm::SmallPtrSet<NodeRef, 4> &Set) {
  return nodesBetweenImpl<revng::detail::FilterSet::WhiteList,
                          GraphT,
                          llvm::GraphTraits<GraphT>>(Source, Destination, Set);
}

template<class GraphT,
         typename NodeRef = typename llvm::GraphTraits<GraphT>::NodeRef>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetweenBlackList(GraphT Source,
                      GraphT Destination,
                      llvm::SmallPtrSet<NodeRef, 4> &Set) {
  return nodesBetweenImpl<revng::detail::FilterSet::BlackList,
                          GraphT,
                          llvm::GraphTraits<GraphT>>(Source, Destination, Set);
}

template<class GraphT>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetweenReverse(GraphT Source, GraphT Destination) {
  using namespace llvm;
  llvm::SmallPtrSet<typename llvm::GraphTraits<GraphT>::NodeRef, 4> EmptySet;
  return nodesBetweenImpl<revng::detail::FilterSet::BlackList,
                          GraphT,
                          GraphTraits<Inverse<GraphT>>>(Source,
                                                        Destination,
                                                        EmptySet);
}

template<
  class GraphT,
  typename NodeRef = typename llvm::GraphTraits<llvm::Inverse<GraphT>>::NodeRef>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetweenReverseWhiteList(GraphT Source,
                             GraphT Destination,
                             llvm::SmallPtrSet<NodeRef, 4> &Set) {
  using namespace llvm;
  return nodesBetweenImpl<revng::detail::FilterSet::WhiteList,
                          GraphT,
                          GraphTraits<Inverse<GraphT>>>(Source,
                                                        Destination,
                                                        Set);
}

template<class GraphT,
         typename NodeRef = typename llvm::GraphTraits<GraphT>::NodeRef>
inline llvm::SmallSetVector<GraphT, 4>
nodesBetweenReverseBlackList(GraphT Source,
                             GraphT Destination,
                             llvm::SmallPtrSet<NodeRef, 4> &Set) {
  using namespace llvm;
  return nodesBetweenImpl<revng::detail::FilterSet::BlackList,
                          GraphT,
                          GraphTraits<Inverse<GraphT>>>(Source,
                                                        Destination,
                                                        Set);
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
