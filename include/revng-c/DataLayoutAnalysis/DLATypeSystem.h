#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <compare>
#include <cstddef>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace dla {

/// Class used to mark InstanceLinkTags between LayoutTypes
struct OffsetExpression {
  uint64_t Offset;
  llvm::SmallVector<uint64_t, 4> Strides;
  llvm::SmallVector<std::optional<uint64_t>, 4> TripCounts;

  explicit OffsetExpression() : OffsetExpression(0ULL){};
  explicit OffsetExpression(uint64_t Off) :
    Offset(Off), Strides(), TripCounts() {}

  OffsetExpression(const OffsetExpression &) = default;
  OffsetExpression &operator=(const OffsetExpression &) = default;

  OffsetExpression(OffsetExpression &&) = default;
  OffsetExpression &operator=(OffsetExpression &&) = default;

  std::strong_ordering
  operator<=>(const OffsetExpression &Other) const = default;

  void print(llvm::raw_ostream &OS) const;

  bool verify() const debug_function {
    if (Offset < 0)
      return false;

    if (Strides.size() != TripCounts.size())
      return false;

    uint64_t PrevStride = std::numeric_limits<uint64_t>::max();
    for (const auto &[Stride, MaybeTC] : llvm::zip_first(Strides, TripCounts)) {

      // Strides should go from larger to smaller
      if (PrevStride < Stride)
        return false;

      // Arrays with unknown length are considered as if they had one element
      auto TripCount = MaybeTC.value_or(1);
      // If the current stride times the current trip count is larger than the
      // previous stride, it would trip over the element of the outer array.
      if (Stride * TripCount > PrevStride)
        return false;

      PrevStride = Stride;
    }

    return true;
  }

  static OffsetExpression append(OffsetExpression LHS,
                                 const OffsetExpression &RHS) {
    revng_assert(LHS.verify());
    revng_assert(RHS.verify());
    LHS.Offset += RHS.Offset;
    LHS.Strides.append(RHS.Strides);
    LHS.TripCounts.append(RHS.TripCounts);
    revng_assert(LHS.verify());
    return LHS;
  }

}; // end class OffsetExpression

class TypeLinkTag {
public:
  enum LinkKind {
    LK_Equality,
    LK_Instance,
    LK_Pointer,
    LK_All,
  };

  static const char *toString(enum LinkKind K) {
    switch (K) {
    case LK_Equality:
      return "Equality";
    case LK_Instance:
      return "Instance";
    case LK_Pointer:
      return "Pointer";
    case LK_All:
      return "None";
    }
    revng_unreachable();
  }

protected:
  OffsetExpression OE;
  const LinkKind Kind;

  explicit TypeLinkTag(LinkKind K, OffsetExpression &&O) :
    OE(std::move(O)), Kind(K) {}

  // TODO: potentially we are interested in marking TypeLinkTags with some info
  // that allows us to track which step on the type system has created them.
  // However, this is not necessary now, so I'll leave it for when we have
  // identified more clearly if we really need it and why.

public:
  TypeLinkTag() = delete;

  LinkKind getKind() const { return Kind; }

  const OffsetExpression &getOffsetExpr() const {
    revng_assert(getKind() == LK_Instance);
    return OE;
  }

  static TypeLinkTag equalityTag() {
    return TypeLinkTag(LK_Equality, OffsetExpression{});
  }

  // This method is templated just to enable perfect forwarding.
  template<typename OffsetExpressionT>
  static TypeLinkTag instanceTag(OffsetExpressionT &&O) {
    return TypeLinkTag(LK_Instance, std::forward<OffsetExpressionT>(O));
  }

  static TypeLinkTag pointerTag() {
    return TypeLinkTag(LK_Pointer, OffsetExpression{});
  }

  std::strong_ordering operator<=>(const TypeLinkTag &Other) const = default;

  friend void
  writeToLog(Logger<true> &L, const dla::TypeLinkTag &T, int /* Ignore */);

}; // end class TypeLinkTag

class LayoutTypeSystem;

enum InterferingChildrenInfo {
  Unknown = 0,
  AllChildrenAreInterfering,
  AllChildrenAreNonInterfering,
};

struct LayoutTypeSystemNode {
  const uint64_t ID = 0ULL;
  using Link = std::pair<LayoutTypeSystemNode *, const TypeLinkTag *>;
  using NeighborsSet = std::set<Link>;
  using NeighborIterator = NeighborsSet::iterator;
  NeighborsSet Successors{};
  NeighborsSet Predecessors{};
  uint64_t Size{};
  InterferingChildrenInfo InterferingInfo{ Unknown };
  LayoutTypeSystemNode(uint64_t I) : ID(I) {}

public:
  // This method should never be called, but it's necessary to be able to use
  // some llvm::GraphTraits algorithms, otherwise they wouldn't compile.
  LayoutTypeSystem *getParent() {
    revng_unreachable();
    return nullptr;
  }

  void print(llvm::raw_ostream &OS) const;

  void printAsOperand(llvm::raw_ostream &OS, bool /* unused */) const {
    print(OS);
  }
};

/// This class handles equivalence classes between indexes of vectors
class VectEqClasses : public llvm::IntEqClasses {
private:
  // ID of the first removed ID
  std::optional<unsigned> RemovedID = {};
  unsigned NElems = 0;

private:
  /// Used internally, operator[] is removed for this class
  unsigned lookupEqClass(unsigned ID) const {
    return llvm::IntEqClasses::operator[](ID);
  }

public:
  /// Add 1 element with its own equivalence class
  unsigned growBy1();

  /// Remove the whole equivalence class of \a ID
  void remove(const unsigned ID);

  /// Check if the element has been removed
  bool isRemoved(const unsigned ID) const;

  /// Get the total number of elements added
  unsigned getNumElements() const { return NElems; }

public:
  /// You can't access the Eq Classes directly, some might be deleted
  unsigned operator[](unsigned) const = delete;

  /// Get the Equivalence class ID of an element (must be compressed)
  ///\return empty if the element is out-of-bounds or has been removed
  std::optional<unsigned> getEqClassID(const unsigned ID) const;

  /// Get all the elements that are in the same equivalence class of \a ID
  ///\note Expensive: performs a linear scan of all the elements
  std::vector<unsigned> computeEqClass(const unsigned ID) const;

  /// Check if \a ID1 and \a ID2 have the same equivalence class
  bool haveSameEqClass(unsigned ID1, unsigned ID2) const;
};

/// This class is used to print debug information about the TypeSystem
///
/// Override this to obtain implementation-specific debug prints.
struct TSDebugPrinter {
  virtual void printNodeContent(const LayoutTypeSystem &TS,
                                const LayoutTypeSystemNode *N,
                                llvm::raw_fd_ostream &File) const;

  virtual ~TSDebugPrinter() {}
};

class LayoutTypeSystem {
public:
  using Node = LayoutTypeSystemNode;
  using NodePtr = LayoutTypeSystemNode *;
  using NodeUniquePtr = std::unique_ptr<LayoutTypeSystemNode>;
  using NeighborIterator = LayoutTypeSystemNode::NeighborIterator;

  LayoutTypeSystem() : DebugPrinter(new TSDebugPrinter) {}

  ~LayoutTypeSystem() {
    for (auto *Layout : Layouts) {
      Layout->~LayoutTypeSystemNode();
      NodeAllocator.Deallocate(Layout);
    }
    Layouts.clear();
  }

public:
  LayoutTypeSystemNode *createArtificialLayoutType();

  llvm::SmallVector<LayoutTypeSystemNode *, 2>
  createArtificialLayoutTypes(unsigned N);

protected:
  // This method is templated only to enable perfect forwarding.
  template<typename TagT>
  std::pair<const TypeLinkTag *, bool>
  addLink(LayoutTypeSystemNode *Src, LayoutTypeSystemNode *Tgt, TagT &&Tag) {
    if (Src == nullptr or Tgt == nullptr or Src == Tgt)
      return std::make_pair(nullptr, false);
    revng_assert(Layouts.contains(Src));
    revng_assert(Layouts.contains(Tgt));
    auto It = LinkTags.insert(std::forward<TagT>(Tag)).first;
    revng_assert(It != LinkTags.end());
    const TypeLinkTag *T = &*It;
    bool New = Src->Successors.insert(std::make_pair(Tgt, T)).second;
    New |= Tgt->Predecessors.insert(std::make_pair(Src, T)).second;
    return std::make_pair(T, New);
  }

public:
  std::pair<const TypeLinkTag *, bool>
  addEqualityLink(LayoutTypeSystemNode *Src, LayoutTypeSystemNode *Tgt) {
    auto ForwardLinkTag = addLink(Src, Tgt, dla::TypeLinkTag::equalityTag());
    auto BackwardLinkTag = addLink(Tgt, Src, dla::TypeLinkTag::equalityTag());
    revng_assert(ForwardLinkTag == BackwardLinkTag);
    return ForwardLinkTag;
  }

  // This method is templated just to enable perfect forwarding.
  template<typename OffsetExpressionT>
  std::pair<const TypeLinkTag *, bool>
  addInstanceLink(LayoutTypeSystemNode *Src,
                  LayoutTypeSystemNode *Tgt,
                  OffsetExpressionT &&OE) {
    using OET = OffsetExpressionT;

    // HACK: If the offset is greater than 64K, avoid adding the link. This
    // avoids creating gigantic structs with a big leading padding in cases in
    // which the DLA was not able to recognize a constant as the base address.
    // In the future, this should be handled through segments.
    if (OE.Offset > 0xFFFF)
      return std::make_pair(nullptr, false);

    return addLink(Src,
                   Tgt,
                   dla::TypeLinkTag::instanceTag(std::forward<OET>(OE)));
  }

  std::pair<const TypeLinkTag *, bool>
  addPointerLink(LayoutTypeSystemNode *Src, LayoutTypeSystemNode *Tgt) {
    return addLink(Src, Tgt, dla::TypeLinkTag::pointerTag());
  }

  void dumpDotOnFile(const char *FName,
                     bool ShowCollapsed = false) const debug_function;

  void dumpDotOnFile(const std::string &FName,
                     bool ShowCollapsed = false) const {
    dumpDotOnFile(FName.c_str(), ShowCollapsed);
  }

  auto getNumLayouts() const { return Layouts.size(); }

  auto getLayoutsRange() const {
    return llvm::make_range(Layouts.begin(), Layouts.end());
  }

public:
  void mergeNodes(const std::vector<LayoutTypeSystemNode *> &ToMerge);

  void removeNode(LayoutTypeSystemNode *N);

  void moveEdgeTarget(LayoutTypeSystemNode *OldTgt,
                      LayoutTypeSystemNode *NewTgt,
                      NeighborIterator InverseEdgeIt,
                      int64_t OffsetToSum);

  void moveEdgeSource(LayoutTypeSystemNode *OldSrc,
                      LayoutTypeSystemNode *NewSrc,
                      NeighborIterator EdgeIt,
                      int64_t OffsetToSum);

  NeighborIterator eraseEdge(LayoutTypeSystemNode *Src,
                             NeighborIterator EdgeIt);

  void dropOutgoingEdges(LayoutTypeSystemNode *N);

private:
  uint64_t NID = 0ULL;

  // Holds all the LayoutTypeSystemNode
  llvm::BumpPtrAllocator NodeAllocator = {};
  std::set<LayoutTypeSystemNode *> Layouts = {};

  // Holds the link tags, so that they can be deduplicated and referred to using
  // TypeLinkTag * in the links inside LayoutTypeSystemNode
  std::set<TypeLinkTag> LinkTags = {};

public:
  // Checks that is valid, and returns true if it is, false otherwise
  bool verifyConsistency() const;
  // Checks that is valid and a DAG, and returns true if it is, false otherwise
  bool verifyDAG() const;
  // Checks that is valid and a DAG on instance. Returns true on success.
  bool verifyInstanceDAG() const;
  // Checks that is valid and a DAG on pointer. Returns true on success.
  bool verifyPointerDAG() const;
  // Checks that there are no leaf nodes without valid layout information
  bool verifyLeafs() const;
  // Checks that there are no equality edges.
  bool verifyNoEquality() const;
  // Checks that there are no equality edges.
  bool verifyInstanceAtOffset0DAG() const;
  // Checks that no union node has only one child
  bool verifyUnions() const;

private:
  // Equivalence classes between nodes. Each node is identified by an ID.
  VectEqClasses EqClasses;
  // Object that defines how the content of each node should be printed
  std::unique_ptr<TSDebugPrinter> DebugPrinter;

public:
  unsigned getNID() const { return NID; }

  VectEqClasses &getEqClasses() { return EqClasses; }
  const VectEqClasses &getEqClasses() const { return EqClasses; }

  void setDebugPrinter(std::unique_ptr<TSDebugPrinter> &&Printer) {
    DebugPrinter = std::move(Printer);
  }
}; // end class LayoutTypeSystem

} // end namespace dla

template<>
struct llvm::GraphTraits<dla::LayoutTypeSystemNode *> {
protected:
  using NodeT = dla::LayoutTypeSystemNode;

public:
  using NodeRef = NodeT *;
  using EdgeRef = const NodeT::NeighborsSet::value_type;

  static NodeRef edge_dest(EdgeRef E) { return E.first; }
  using EdgeDestT = NodeRef (*)(EdgeRef);

  using ChildEdgeIteratorType = NodeT::NeighborsSet::iterator;
  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  EdgeDestT>;

  static NodeRef getEntryNode(const NodeRef &N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(N->Successors.begin(), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(N->Successors.end(), edge_dest);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->Successors.begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->Successors.end();
  }
}; // end struct llvm::GraphTraits<dla::LayoutTypeSystemNode *>

template<>
struct llvm::GraphTraits<const dla::LayoutTypeSystemNode *> {
protected:
  using NodeT = const dla::LayoutTypeSystemNode;

public:
  using NodeRef = NodeT *;
  using EdgeRef = const NodeT::NeighborsSet::value_type;

  static NodeRef edge_dest(EdgeRef E) { return E.first; }
  using EdgeDestT = NodeRef (*)(EdgeRef);

  using ChildEdgeIteratorType = NodeT::NeighborsSet::iterator;
  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  EdgeDestT>;

  static NodeRef getEntryNode(const NodeRef &N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(N->Successors.begin(), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(N->Successors.end(), edge_dest);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->Successors.begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->Successors.end();
  }
}; // end struct llvm::GraphTraits<dla::LayoutTypeSystemNode *>

template<>
struct llvm::GraphTraits<llvm::Inverse<dla::LayoutTypeSystemNode *>> {
protected:
  using NodeT = dla::LayoutTypeSystemNode;

public:
  using NodeRef = NodeT *;
  using EdgeRef = const NodeT::NeighborsSet::value_type;

  static NodeRef edge_dest(EdgeRef E) { return E.first; }
  using EdgeDestT = NodeRef (*)(EdgeRef);

  using ChildEdgeIteratorType = NodeT::NeighborsSet::iterator;
  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  EdgeDestT>;

  static NodeRef getEntryNode(const NodeRef &N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(N->Predecessors.begin(), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(N->Predecessors.end(), edge_dest);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->Predecessors.begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->Predecessors.end();
  }
}; // end struct llvm::GraphTraits<dla::LayoutTypeSystemNode *>

template<>
struct llvm::GraphTraits<llvm::Inverse<const dla::LayoutTypeSystemNode *>> {
protected:
  using NodeT = const dla::LayoutTypeSystemNode;

public:
  using NodeRef = NodeT *;
  using EdgeRef = const NodeT::NeighborsSet::value_type;

  static NodeRef edge_dest(EdgeRef E) { return E.first; }
  using EdgeDestT = NodeRef (*)(EdgeRef);

  using ChildEdgeIteratorType = NodeT::NeighborsSet::iterator;
  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  EdgeDestT>;

  static NodeRef getEntryNode(const NodeRef &N) { return N; }

  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(N->Predecessors.begin(), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(N->Predecessors.end(), edge_dest);
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->Predecessors.begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->Predecessors.end();
  }
}; // end struct llvm::GraphTraits<dla::LayoutTypeSystemNode *>

template<>
struct llvm::GraphTraits<const dla::LayoutTypeSystem *>
  : public llvm::GraphTraits<const dla::LayoutTypeSystemNode *> {

public:
  using nodes_iterator = std::set<dla::LayoutTypeSystemNode *>::iterator;

  static NodeRef getEntryNode(const dla::LayoutTypeSystem *) { return nullptr; }

  static nodes_iterator nodes_begin(const dla::LayoutTypeSystem *G) {
    return G->getLayoutsRange().begin();
  }

  static nodes_iterator nodes_end(const dla::LayoutTypeSystem *G) {
    return G->getLayoutsRange().end();
  }

  static unsigned size(const dla::LayoutTypeSystem *G) {
    return G->getNumLayouts();
  }
}; // struct llvm::GraphTraits<dla::LayoutTypeSystem>

template<>
struct llvm::GraphTraits<dla::LayoutTypeSystem *>
  : public llvm::GraphTraits<dla::LayoutTypeSystemNode *> {

public:
  using nodes_iterator = std::set<dla::LayoutTypeSystemNode *>::iterator;

  static NodeRef getEntryNode(const dla::LayoutTypeSystem *) { return nullptr; }

  static nodes_iterator nodes_begin(const dla::LayoutTypeSystem *G) {
    return G->getLayoutsRange().begin();
  }

  static nodes_iterator nodes_end(const dla::LayoutTypeSystem *G) {
    return G->getLayoutsRange().end();
  }

  static unsigned size(dla::LayoutTypeSystem *G) { return G->getNumLayouts(); }
}; // struct llvm::GraphTraits<dla::LayoutTypeSystem>

namespace dla {

template<dla::TypeLinkTag::LinkKind K>
inline bool hasLinkKind(const dla::LayoutTypeSystemNode::Link &L) {
  if constexpr (K == dla::TypeLinkTag::LinkKind::LK_All)
    return true;
  else
    return L.second->getKind() == K;
}

template<dla::TypeLinkTag::LinkKind K>
inline bool hasNonPointerLinkKind(const dla::LayoutTypeSystemNode::Link &L) {
  static_assert(K != dla::TypeLinkTag::LinkKind::LK_Pointer);
  if constexpr (K == dla::TypeLinkTag::LinkKind::LK_All)
    return L.second->getKind() != dla::TypeLinkTag::LinkKind::LK_Pointer;
  else
    return L.second->getKind() == K;
}

inline bool
isEqualityEdge(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  return hasLinkKind<TypeLinkTag::LinkKind::LK_Equality>(E);
}

inline bool
isInstanceEdge(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  return hasLinkKind<TypeLinkTag::LinkKind::LK_Instance>(E);
}

inline bool
isInstanceOff0(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  if (not isInstanceEdge(E))
    return false;

  auto &OE = E.second->getOffsetExpr();
  return OE.Offset == 0 and OE.Strides.empty() and OE.TripCounts.empty();
}

inline bool
isStridedInstance(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  if (not isInstanceEdge(E))
    return false;

  auto &OE = E.second->getOffsetExpr();
  return not OE.Strides.empty();
}

inline bool
isInstanceOffNon0(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  return isInstanceEdge(E) and not isInstanceOff0(E);
}

inline bool
isPointerEdge(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  return hasLinkKind<TypeLinkTag::LinkKind::LK_Pointer>(E);
}

inline bool
isNotPointerEdge(const llvm::GraphTraits<LayoutTypeSystemNode *>::EdgeRef &E) {
  return not isPointerEdge(E);
}

inline bool isPointerNode(const LayoutTypeSystemNode *N) {
  return llvm::any_of(N->Successors, isPointerEdge);
}

inline bool isStructNode(const LayoutTypeSystemNode *N) {
  return N->InterferingInfo == AllChildrenAreNonInterfering
         and llvm::any_of(N->Successors, isNotPointerEdge);
}

inline bool isUnionNode(const LayoutTypeSystemNode *N) {
  return N->InterferingInfo == AllChildrenAreInterfering;
}

template<dla::TypeLinkTag::LinkKind K = dla::TypeLinkTag::LinkKind::LK_All>
inline bool isLeaf(const LayoutTypeSystemNode *N) {
  using LTSN = const LayoutTypeSystemNode;
  using GraphNodeT = LTSN *;
  using FilteredNodeT = EdgeFilteredGraph<GraphNodeT, hasNonPointerLinkKind<K>>;
  using GT = llvm::GraphTraits<FilteredNodeT>;
  return GT::child_begin(N) == GT::child_end(N);
}

inline bool isInstanceLeaf(const LayoutTypeSystemNode *N) {
  return isLeaf<dla::TypeLinkTag::LinkKind::LK_Instance>(N);
}

inline bool isPointerLeaf(const LayoutTypeSystemNode *N) {
  using LTSN = LayoutTypeSystemNode;
  using PointerNodeT = EdgeFilteredGraph<const LTSN *, isPointerEdge>;
  using PointerGraph = llvm::GraphTraits<PointerNodeT>;
  return PointerGraph::child_begin(N) == PointerGraph::child_end(N);
}

template<dla::TypeLinkTag::LinkKind K = dla::TypeLinkTag::LinkKind::LK_All>
inline bool isRoot(const LayoutTypeSystemNode *N) {
  using LTSN = const LayoutTypeSystemNode;
  using GraphNodeT = LTSN *;
  using FilteredNodeT = EdgeFilteredGraph<GraphNodeT, hasNonPointerLinkKind<K>>;
  using IGT = llvm::GraphTraits<llvm::Inverse<FilteredNodeT>>;
  return IGT::child_begin(N) == IGT::child_end(N);
}

inline bool isInstanceRoot(const LayoutTypeSystemNode *N) {
  return isRoot<dla::TypeLinkTag::LinkKind::LK_Instance>(N);
}

inline bool isPointerRoot(const LayoutTypeSystemNode *N) {
  using LTSN = LayoutTypeSystemNode;
  using PointerNodeT = EdgeFilteredGraph<const LTSN *, isPointerEdge>;
  using PointerGraph = llvm::GraphTraits<llvm::Inverse<PointerNodeT>>;
  return PointerGraph::child_begin(N) == PointerGraph::child_end(N);
}
} // end namespace dla
