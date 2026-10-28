//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/SmallMap.h"

#include "DLAStep.h"
#include "FieldSizeComputation.h"

namespace dla {

struct ChildrenKey {
  uint64_t Size;
  const TypeLinkTag *Tag = nullptr;

  /// Compares the tags through their contents. A defaulted comparison would
  /// compare the pointers, and the tags are allocated wherever the type
  /// system put them. Contents are as good as identity here, since the type
  /// system keeps its tags in a `std::set<TypeLinkTag>`, so two tags with the
  /// same contents are the same object.
  std::strong_ordering operator<=>(const ChildrenKey &Other) const noexcept {
    if (auto Cmp = Size <=> Other.Size; Cmp != std::strong_ordering::equal)
      return Cmp;

    if (Tag == nullptr or Other.Tag == nullptr)
      return (Tag != nullptr) <=> (Other.Tag != nullptr);

    return *Tag <=> *Other.Tag;
  }

  bool operator==(const ChildrenKey &Other) const noexcept {
    return (*this <=> Other) == std::strong_ordering::equal;
  }
};

using NodePredicate = const std::function<bool(const LayoutTypeSystemNode *)>;

using NeighborIterator = LayoutTypeSystemNode::NeighborIterator;

/// Orders two edges by the ID of the child and by the contents of the tag.
///
/// `resolveUnion` walks the set below pairwise and erases the less general of
/// every comparable pair, and "more general" is not a total order, so which
/// children survive depends on the order they are visited in. Ordering by the
/// addresses of the nodes and of the tags would hand that decision to the
/// allocator.
static bool neighborLess(const NeighborIterator &AIt,
                         const NeighborIterator &BIt) {
  const auto &[AChild, ATag] = *AIt;
  const auto &[BChild, BTag] = *BIt;

  if (AChild->ID != BChild->ID)
    return AChild->ID < BChild->ID;

  if (ATag == nullptr or BTag == nullptr)
    return ATag == nullptr and BTag != nullptr;

  return *ATag < *BTag;
}

using NeighborLess = std::integral_constant<decltype(&neighborLess),
                                            neighborLess>;

using NeighborSet = std::set<NeighborIterator, NeighborLess>;

static SmallMap<ChildrenKey, NeighborSet, 8>
getOverlappingLeafChildren(LayoutTypeSystemNode *N) {

  SmallMap<ChildrenKey, NeighborSet, 8> Result;

  auto ChildIt = N->Successors.begin();
  auto ChildEnd = N->Successors.end();
  for (; ChildIt != ChildEnd; ++ChildIt) {
    if (not isInstanceEdge(*ChildIt))
      continue;
    auto &[Child, Tag] = *ChildIt;
    if (not isLeaf(Child))
      continue;
    Result[ChildrenKey{ getFieldSize(Child, Tag), Tag }].insert(ChildIt);
  }

  return Result;
}

using GT = llvm::GraphTraits<LayoutTypeSystemNode *>;

static bool isInstanceAtOffset0(const GT::EdgeRef &E) {
  if (not isInstanceEdge(E))
    return false;

  return not E.second->getOffsetExpr().Offset;
}

// Returns greater if the type represented by A transitively contains an
// instance at offset 0 of the type represented by B.
// Returns less if the type represented by B transitively contains an
// instance at offset 0 of the type represented by A.
// Returns equivalent if A == B.
// Returns unordered in all the other cases.
static std::partial_ordering comparePointee(const LayoutTypeSystemNode *A,
                                            const LayoutTypeSystemNode *B) {
  using CInstance0 = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                       isInstanceAtOffset0>;
  using NodeSet = llvm::df_iterator_default_set<const LayoutTypeSystemNode *,
                                                8>;

  if (A == B)
    return std::partial_ordering::equivalent;

  NodeSet Visited;
  for (const LayoutTypeSystemNode *N :
       llvm::depth_first_ext(CInstance0(A), Visited))
    if (N == B)
      return std::partial_ordering::greater;

  for (const LayoutTypeSystemNode *N :
       llvm::depth_first_ext(CInstance0(B), Visited))
    if (N == A)
      return std::partial_ordering::less;

  return std::partial_ordering::unordered;
}

// TODO: One day, when we teach DLA about VMA and model types, we should support
// all model::PrimitiveTypes here.
struct LeafType {
  enum Kind {
    Generic,
    Pointer
  } Kind;
  // The following should be nullptr if Kind is not Pointer
  const LayoutTypeSystemNode *Pointee = nullptr;
};

static LeafType getType(const LayoutTypeSystemNode *N) {
  if (isPointerNode(N)) {
    revng_assert(N->Successors.size() == 1);
    const LayoutTypeSystemNode *Pointee = N->Successors.begin()->first;
    return LeafType{ LeafType::Pointer, Pointee };
  }
  return LeafType{ LeafType::Generic, nullptr };
}

// Returns:
// - equivalent if the types represented by A and B are the same
// - greater if the type represented by A can model more traversals on the DLA
//   type graph than that represented by B
// - less if the type represented by B can model mode traversals on the DLA type
//   graph than that represented by A
// - unordered otherwise
static std::partial_ordering compareLeafTypes(const LayoutTypeSystemNode *A,
                                              const LayoutTypeSystemNode *B) {
  LeafType AType = getType(A);
  LeafType BType = getType(B);

  switch (AType.Kind) {

  case LeafType::Generic: {

    switch (BType.Kind) {

    case LeafType::Generic: {
      return std::partial_ordering::equivalent;
    }

    case LeafType::Pointer: {
      return std::partial_ordering::less;
    }

    default:
      revng_abort();
    }

  } break;

  case LeafType::Pointer: {

    switch (BType.Kind) {

    case LeafType::Generic: {
      return std::partial_ordering::greater;
    }

    case LeafType::Pointer: {
      return comparePointee(AType.Pointee, BType.Pointee);
    }

    default:
      revng_abort();
    }

  } break;

  default:
    revng_abort();
  }

  return std::partial_ordering::unordered;
}

// Whenever in ChildrenSet there are multiple pointer children that point to
// types that are one at offset 0 of the other, it removes from Parent the
// pointer children that point to the less-general types.
static bool resolveUnion(LayoutTypeSystem &TS,
                         LayoutTypeSystemNode *Parent,
                         NeighborSet &ChildrenSet) {
  bool Changed = false;

  auto End = ChildrenSet.end();
  auto AIt = ChildrenSet.begin();
  auto ANext = AIt;

  for (; AIt != End; AIt = ANext) {
    ANext = std::next(AIt);

    auto AChildIt = *AIt;
    LayoutTypeSystemNode *AChild = AChildIt->first;

    auto BIt = ANext;
    auto BNext = BIt;
    for (; BIt != End; BIt = BNext) {
      BNext = std::next(BIt);

      auto BChildIt = *BIt;
      LayoutTypeSystemNode *BChild = BChildIt->first;

      auto Cmp = compareLeafTypes(AChild, BChild);
      // A can reach more types on the DLA graph than B.
      // Remove B.
      if (Cmp > 0) {
        BNext = ChildrenSet.erase(BIt);
        if (ANext == BIt)
          ANext = std::next(AIt);
        TS.eraseEdge(Parent, BChildIt);
        Changed = true;
      }

      // B can reach more types on the DLA graph than A.
      // Remove A.
      if (Cmp < 0) {
        ANext = ChildrenSet.erase(AIt);
        TS.eraseEdge(Parent, AChildIt);
        Changed = true;
        break;
      }
    }
  }
  // TODO: should we remove some nodes altogether? Or should we merge the
  // nodes that are killed with the nodes that survive?
  return Changed;
}

bool ResolveLeafUnions::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  for (LayoutTypeSystemNode *Node : llvm::nodes(&TS)) {
    auto LeafChildrenSets = getOverlappingLeafChildren(Node);
    for (auto &LeafChildrenSet : llvm::make_second_range(LeafChildrenSets))
      Changed |= resolveUnion(TS, Node, LeafChildrenSet);
  }

  return Changed;
}

} // end namespace dla
