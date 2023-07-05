/// \file LinearSegmentSelection.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <deque>

#include "llvm/ADT/PostOrderIterator.h"

#include "InternalCompute.h"

/// A helper used to simplify the parent lookup when deciding on the linear
/// segments of the layout.
class ParentLookupHelper {
protected:
  NodeView lookupImpl(NodeView Node) {
    if (auto Iterator = HeadsRef.find(Node); Iterator == HeadsRef.end())
      return (HeadsRef[Node] = Node);
    else if (Iterator->second == Node)
      return Node;
    else
      return lookupImpl(Iterator->second);
  }

public:
  ParentLookupHelper(SegmentContainer &HeadsRef) : HeadsRef(HeadsRef) {}
  NodeView operator()(NodeView Node) { return lookupImpl(Node); }

private:
  SegmentContainer &HeadsRef;
};

/// \note: it's probably a good idea to split this function up into a couple
/// of smaller ones, it's too long.
SegmentContainer selectLinearSegments(InternalGraph &Graph,
                                      const RankContainer &Ranks,
                                      const LayerContainer &Layers,
                                      const std::vector<NodeView> &Order) {
  SegmentContainer Heads, Tails;

  std::vector<NodeView> Modified;
  for (auto *Node : Graph.nodes()) {
    Heads[Node] = Tails[Node] = Node;
    Modified.emplace_back(Node);
  }

  while (!Modified.empty()) {
    decltype(Modified) NewlyModified;
    for (auto Node : Modified) {
      if (Tails.at(Node)->successorCount() == 1) {
        auto Current = Tails.at(Node);
        auto *Successor = *Current->successors().begin();
        if (Ranks.at(Current) < Ranks.at(Successor)) {
          if (Successor->predecessorCount() == 1) {
            // Merge segments
            Tails[Node] = Tails.at(Successor);
            NewlyModified.push_back(Node);
          }
        }
      }
    }

    std::swap(Modified, NewlyModified);
  }

  ParentLookupHelper ParentLookup(Heads);
  for (auto &[Head, Tail] : Tails) {
    auto ParentView = ParentLookup(Head);

    auto Current = Head;
    while (Current != Tail) {
      revng_assert(Current->hasSuccessors());
      Current = *Current->successors().begin();
      Heads[ParentLookup(Current)] = ParentView;
    }
  }

  auto &ParentMap = Heads;

  std::unordered_map<NodeView, size_t> OrderLookupTable;
  for (size_t Index = 0; Index < Order.size(); ++Index)
    OrderLookupTable[Order[Index]] = Index;

  std::deque<NodeView> Queue;
  for (auto &[Node, Parent] : ParentMap)
    if (Node != ParentLookup(Parent))
      Queue.emplace_back(Node);

  while (!Queue.empty()) {
    auto Current = Queue.front();
    Queue.pop_front();

    auto Parent = ParentMap[Current];
    if (Current == Parent)
      continue;

    auto CurrentIndex = OrderLookupTable[Current];
    auto ParentIndex = OrderLookupTable[Parent];
    for (auto Cousin : Layers.at(Ranks.at(Current))) {
      if (Cousin == Current)
        continue;

      auto CousinIndex = OrderLookupTable[Cousin];
      if (ParentMap.contains(Cousin)) {
        if (auto Ommer = ParentMap[Cousin]; Ommer != Cousin) {
          auto OmmerIndex = OrderLookupTable[Ommer];
          auto PD = RankDelta(ParentIndex) - RankDelta(OmmerIndex);
          auto CD = RankDelta(CurrentIndex) - RankDelta(CousinIndex);
          // Compares the relative directions between parents and children.
          // If `ParentIndex < OmmerIndex` (the parent is placed before the
          // ommer), the `PD` is negative. It's the same for children (`CD`).
          // Multiplication acts as a `XOR` operation on them - if both
          // relations are the same - the product is positive. It's negative
          // otherwise. The straightforward segmentation is impossible for
          // the case with opposite orientation, that's why the segments are
          // broken (both `Current` and `Cousin` are set as the heads of their
          // respective segments).
          if (PD * CD < 0) {
            ParentMap[Current] = Current;
            ParentMap[Cousin] = Cousin;

            for (auto *Successor : Current->successors())
              Queue.emplace_back(Successor);
            for (auto *Successor : Cousin->successors())
              Queue.emplace_back(Successor);
          }
        }
      }
    }
  }

  SegmentContainer Result;
  for (auto &[Node, _] : ParentMap)
    Result[Node] = ParentLookup(Node);

  if (Result.size() != Graph.size())
    for (auto *Node : Graph.nodes())
      Result.try_emplace(Node, Node);
  revng_assert(Result.size() == Graph.size());

  return Result;
}

SegmentContainer emptyLinearSegments(InternalGraph &Graph) {
  SegmentContainer Result;

  for (auto *Node : Graph.nodes())
    Result[Node] = Node;

  return Result;
}
