#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Debug.h"

/// GenericGraph is our implementation of a universal graph architecture.
/// First and foremost, it's tailored for the revng codebase, but maybe you
/// can find it useful as well, that's why it's released under the MIT License.
///
/// To use the Graph, you need to make you choice of the node type. We're
/// currently supporting three architectures:
///
///   - ForwardNode - a trivially-simple single-linked node. It uses the least
///     memory, because each node only stores the list of its successors.
///     This makes backwards iteration impossible without the reference to
///     the whole graph (and really expensive even in those cases).
///
///   - BidirectionalNode - a simple double-linked node. It stores the both
///     lists of its successors and predecessors. Take note, that for the sake
///     of implementation simplicity this node stores COPIES of the labels.
///     It's only suitable to be used with cheap to copy labels which are never
///     mutated. There are plans on making them explicitly immutable (TODO),
///     as such you can consider label mutation a deprecated behavior.
///
///   - MutableEdgeNode - a double-linked node with dynamically allocated
///     labels. It is similar to BidirectionalNode except it stores edge labels
///     on the heap. It's slower and uses more memory but allows for safe label
///     modification as well as controls that nodes and edges are removed
///     safely, with the other "halves" cleaned up as well.
///     Note that it's disallowed to have a mutable edge graph without edge
///     labels. Use `BidirectionalNode` in those cases.
///
///   - The node you're going to write - No-one knows the needs of your
///     project better than you. That's why the best data structure is the one
///     you are going to write. So just inherit one of our nodes, or even copy
///     it and modify it so that it suits your graphs as nicely as possible.
///
///  On the side-note, we're providing a couple of helpful concepts to help
///  differentiate different node types. This is helpful in the projects that
///  use multiple different graph architectures side by side.

template<typename T>
concept SpecializationOfForwardNode = requires { T::is_forward_node; };

template<typename T>
concept StrictSpecializationOfBidirectionalNode = requires {
  T::is_bidirectional_node;
  typename llvm::Inverse<T *>;
};

template<typename T>
concept StrictSpecializationOfMutableEdgeNode = requires {
  T::is_mutable_edge_node;
  typename llvm::Inverse<T *>;
};

template<typename T>
concept SpecializationOfGenericGraph = requires {
  T::is_generic_graph;
  typename T::Node;
};

struct Empty {
  bool operator==(const Empty &) const { return true; }
};

template<typename Node, size_t SmallSize = 16, bool HasEntryNode = true>
class GenericGraph;

template<typename T, typename BaseType>
class Parent : public BaseType {
public:
  template<typename... ArgTypes>
  explicit Parent(ArgTypes &&...Args) :
    BaseType(std::forward<ArgTypes>(Args)...), TheParent(nullptr) {}

  Parent(const Parent &) = default;
  Parent(Parent &&) = default;

private:
  T *TheParent;

public:
  T *getParent() const { return TheParent; }
  void setParent(T *Parent) { this->TheParent = Parent; }
};

/// Data structure for edge labels
template<typename Node, typename EdgeLabel>
struct Edge : public EdgeLabel {
  Edge(Node *Neighbor) : Neighbor(Neighbor) {}
  Edge(Node *Neighbor, EdgeLabel EL) : EdgeLabel(EL), Neighbor(Neighbor) {}
  Node *Neighbor = nullptr;
  bool operator==(const Edge &Other) const {
    return Other.Neighbor == Neighbor and EdgeLabel::operator==(Other);
  }
};

//
// Structured binding support for Edge
//
template<size_t I, typename Node, typename EdgeLabel>
auto &get(Edge<Node, EdgeLabel> &E) {
  if constexpr (I == 0)
    return E.Neighbor;
  else
    return static_cast<EdgeLabel &>(E);
}

template<size_t I, typename Node, typename EdgeLabel>
auto &get(const Edge<Node, EdgeLabel> &E) {
  if constexpr (I == 0)
    return E.Neighbor;
  else
    return static_cast<EdgeLabel &>(E);
}

template<size_t I, typename Node, typename EdgeLabel>
auto &&get(Edge<Node, EdgeLabel> &&E) {
  if constexpr (I == 0)
    return std::move(E.Neighbor);
  else
    return std::move(static_cast<EdgeLabel &&>(E));
}

namespace std {
template<typename Node, typename EdgeLabel>
struct std::tuple_size<Edge<Node, EdgeLabel>>
  : std::integral_constant<size_t, 2> {};

template<typename Node, typename EdgeLabel>
struct std::tuple_element<0, Edge<Node, EdgeLabel>> {
  using type = Node *;
};

template<typename Node, typename EdgeLabel>
struct std::tuple_element<1, Edge<Node, EdgeLabel>> {
  using type = EdgeLabel;
};

} // namespace std

namespace revng::detail {
/// We require to operate some decision to select a base type that Forward node
/// will extend. Those decisions are wrapped inside this struct to remove
/// clutter.
///
/// \note At this step Forward edge has not been declared yet, thus we accept a
///       template parameter that has the same signature as ForwardEdge that
///       will be declared later. This allows us to use it as if it was
///       declared, provided that only the real ForwardEdge is used as this
///       argument.
template<typename Node,
         typename EdgeLabel,
         bool NeedsParent,
         size_t SmallSize,
         template<typename, typename, bool, size_t, typename, size_t, bool>
         class ForwardEdge,
         typename FinalType,
         size_t ParentSmallSize,
         bool ParentHasEntryNode>
struct ForwardNodeBaseTCalc {
  static constexpr bool
    NoDerivation = std::is_same_v<FinalType, std::false_type>;
  using FWNode = ForwardEdge<Node,
                             EdgeLabel,
                             NeedsParent,
                             SmallSize,
                             FinalType,
                             ParentSmallSize,
                             ParentHasEntryNode>;
  using DerivedType = std::conditional_t<NoDerivation, FWNode, FinalType>;
  using GenericGraph = GenericGraph<DerivedType,
                                    ParentSmallSize,
                                    ParentHasEntryNode>;
  using ParentType = Parent<GenericGraph, Node>;
  using Result = std::conditional_t<NeedsParent, ParentType, Node>;
};
} // namespace revng::detail

/// Basic nodes type, only forward edges, possibly with parent
template<typename Node,
         typename EdgeLabel = Empty,
         bool NeedsParent = true,
         size_t SmallSize = 2,
         typename FinalType = std::false_type,
         size_t ParentSmallSize = 16,
         bool ParentHasEntryNode = true>
class ForwardNode
  : public revng::detail::ForwardNodeBaseTCalc<Node,
                                               EdgeLabel,
                                               NeedsParent,
                                               SmallSize,
                                               ForwardNode,
                                               FinalType,
                                               ParentSmallSize,
                                               ParentHasEntryNode>::Result {
public:
  static constexpr bool is_forward_node = true;
  static constexpr bool HasParent = NeedsParent;
  using TypeCalc = revng::detail::ForwardNodeBaseTCalc<Node,
                                                       EdgeLabel,
                                                       NeedsParent,
                                                       SmallSize,
                                                       ForwardNode,
                                                       FinalType,
                                                       ParentSmallSize,
                                                       ParentHasEntryNode>;
  using DerivedType = typename TypeCalc::DerivedType;
  using Base = typename TypeCalc::Result;
  using Edge = Edge<DerivedType, EdgeLabel>;
  using NodeData = Node;

public:
  template<typename... ArgTypes>
  explicit ForwardNode(ArgTypes &&...Args) :
    Base(std::forward<ArgTypes>(Args)...) {}

  ForwardNode(const ForwardNode &) = default;
  ForwardNode(ForwardNode &&) = default;
  ForwardNode &operator=(const ForwardNode &) = default;
  ForwardNode &operator=(ForwardNode &&) = default;

  NodeData &data() { return *this; }
  const NodeData &data() const { return *this; }

  NodeData copyData() const { return *this; }
  NodeData &&moveData() { return std::move(*this); }

public:
  static DerivedType *&getNeighbor(Edge &E) { return E.Neighbor; }
  static const DerivedType *const &getConstNeighbor(const Edge &E) {
    return E.Neighbor;
  }

  static Edge *getEdgePointer(Edge &E) { return &E; }
  static const Edge *getConstEdgePointer(const Edge &E) { return &E; }

private:
  using iterator_filter = decltype(&getNeighbor);
  using const_iterator_filter = decltype(&getConstNeighbor);

public:
  using NeighborContainer = llvm::SmallVector<Edge, SmallSize>;
  using child_iterator = llvm::mapped_iterator<Edge *, iterator_filter>;
  using const_child_iterator = llvm::mapped_iterator<const Edge *,
                                                     const_iterator_filter>;
  using edge_iterator = typename NeighborContainer::iterator;
  using const_edge_iterator = typename NeighborContainer::const_iterator;

public:
  // This stuff is needed by the DominatorTree implementation
  void printAsOperand(llvm::raw_ostream &, bool) const { revng_abort(); }

public:
  void addSuccessor(DerivedType *NewSuccessor) {
    Successors.emplace_back(NewSuccessor);
  }

  void addSuccessor(DerivedType *NewSuccessor, EdgeLabel EL) {
    Successors.emplace_back(NewSuccessor, EL);
  }

public:
  child_iterator removeSuccessor(child_iterator It) {
    auto InternalIt = Successors.erase(It.getCurrent());
    return child_iterator(InternalIt, getNeighbor);
  }

  edge_iterator removeSuccessorEdge(edge_iterator It) {
    return Successors.erase(It);
  }

  void clearSuccessors() { Successors.clear(); }

public:
  llvm::iterator_range<const_child_iterator> successors() const {
    return toNeighborRange(Successors);
  }

  llvm::iterator_range<child_iterator> successors() {
    return toNeighborRange(Successors);
  }

  llvm::iterator_range<const_edge_iterator> successor_edges() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  llvm::iterator_range<edge_iterator> successor_edges() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  bool hasSuccessors() const { return Successors.size() != 0; }
  size_t successorCount() const { return Successors.size(); }
  bool hasSuccessor(DerivedType const *S) const {
    auto Iterator = llvm::find_if(Successors,
                                  [S](auto &N) { return N.Neighbor == S; });
    return Iterator != Successors.end();
  }

protected:
  static llvm::iterator_range<child_iterator>
  toNeighborRange(NeighborContainer &Neighbors) {
    auto Range = llvm::make_range(Neighbors.begin(), Neighbors.end());
    return llvm::map_range(Range, &getNeighbor);
  }

  static llvm::iterator_range<const_child_iterator>
  toNeighborRange(const NeighborContainer &Neighbors) {
    auto Range = llvm::make_range(Neighbors.begin(), Neighbors.end());
    return llvm::map_range(Range, &getConstNeighbor);
  }

protected:
  NeighborContainer Successors;
};

namespace revng::detail {

/// To remove clutter from BidirectionalNode, the computation of some types are
/// done in this class.
///
/// \note At this step BidirectionalEdge has not been declared yet, thus we
///       accept a template parameter that has the same signature as
///       BidirectionalEdge that will be declared later. This allows us to use
///       it as if it was declared, provided that only the real
///       BidirectionalEdge is used as this argument.
template<typename Node,
         typename EdgeLabel,
         bool NeedsParent,
         size_t SmallSize,
         template<typename, typename, bool, size_t>
         class BidirectionalNode>
struct BidirectionalNodeBaseTCalc {
  using BDNode = BidirectionalNode<Node, EdgeLabel, NeedsParent, SmallSize>;
  using Result = ForwardNode<Node, EdgeLabel, NeedsParent, SmallSize, BDNode>;
};
} // namespace revng::detail

using revng::detail::BidirectionalNodeBaseTCalc;

/// Same as ForwardNode, but with backward links too
/// TODO: Make edge labels immutable
template<typename Node,
         typename EdgeLabel = Empty,
         bool NeedsParent = true,
         size_t SmallSize = 2>
class BidirectionalNode
  : public BidirectionalNodeBaseTCalc<Node,
                                      EdgeLabel,
                                      NeedsParent,
                                      SmallSize,
                                      BidirectionalNode>::Result {
public:
  // NOLINTNEXTLINE
  static const bool is_bidirectional_node = true;

public:
  using NodeData = Node;
  using EdgeLabelData = EdgeLabel;
  using Base = ForwardNode<Node,
                           EdgeLabel,
                           NeedsParent,
                           SmallSize,
                           BidirectionalNode>;
  using NeighborContainer = typename Base::NeighborContainer;
  using child_iterator = typename Base::child_iterator;
  using const_child_iterator = typename Base::const_child_iterator;
  using edge_iterator = typename Base::edge_iterator;
  using const_edge_iterator = typename Base::const_edge_iterator;

public:
  template<typename... ArgTypes>
  explicit BidirectionalNode(ArgTypes &&...Args) :
    Base(std::forward<ArgTypes>(Args)...) {}

  BidirectionalNode(const BidirectionalNode &) = default;
  BidirectionalNode(BidirectionalNode &&) = default;

  NodeData &data() { return *this; }
  const NodeData &data() const { return *this; }

  NodeData copyData() const { return *this; }
  NodeData &&moveData() { return std::move(*this); }

public:
  void addSuccessor(BidirectionalNode *NewSuccessor) {
    Base::Successors.emplace_back(NewSuccessor);
    NewSuccessor->Predecessors.emplace_back(this);
  }

  void addSuccessor(BidirectionalNode *NewSuccessor, EdgeLabel EL) {
    Base::Successors.emplace_back(NewSuccessor, EL);
    NewSuccessor->Predecessors.emplace_back(this, EL);
  }

  void addPredecessor(BidirectionalNode *NewPredecessor) {
    Predecessors.emplace_back(NewPredecessor);
    NewPredecessor->Successors.emplace_back(this);
  }

  void addPredecessor(BidirectionalNode *NewPredecessor, EdgeLabel EL) {
    Predecessors.emplace_back(NewPredecessor, EL);
    NewPredecessor->Successors.emplace_back(this, EL);
  }

public:
  void removePredecessor(child_iterator It) {
    BidirectionalNode *Predecessor = *It;
    bool Found = false;
    auto PredecessorSuccessors = Predecessor->successor_edges();
    for (auto PredecessorIt = PredecessorSuccessors.begin(),
              Last = PredecessorSuccessors.end();
         PredecessorIt != Last;) {
      auto Edge = *PredecessorIt;
      Edge.Neighbor = this;
      if (Edge == *PredecessorIt) {
        Predecessor->Successors.erase(PredecessorIt);
        Found = true;
        break;
      } else {
        ++PredecessorIt;
      }
    }
    revng_assert(Found);

    Predecessors.erase(It.getCurrent());
  }

  void removePredecessorEdge(edge_iterator It) {
    Edge Edge = *It;
    bool Found = false;

    // Extract predecessor
    auto Predecessor = Edge.Neighbor;

    // Invert the edge direction
    Edge.Neighbor = this;

    auto PredecessorSuccessors = Predecessor->successor_edges();
    for (auto PredecessorIt = PredecessorSuccessors.begin(),
              Last = PredecessorSuccessors.end();
         PredecessorIt != Last;) {
      if (Edge == *PredecessorIt) {
        Predecessor->Successors.erase(PredecessorIt);
        Found = true;
        break;
      } else {
        ++PredecessorIt;
      }
    }
    revng_assert(Found);

    Predecessors.erase(It);
  }

  void clearPredecessors() {
    while (Predecessors.size())
      removePredecessorEdge(Predecessors.begin());
  }

  void removeSuccessor(child_iterator It) {
    BidirectionalNode *Successor = *It;
    bool Found = false;
    auto SuccessorPredecessors = Successor->predecessor_edges();
    for (auto SuccessorIt = SuccessorPredecessors.begin(),
              Last = SuccessorPredecessors.end();
         SuccessorIt != Last;) {
      auto Edge = *SuccessorIt;
      Edge.Neighbor = this;
      if (Edge == *SuccessorIt) {
        Successor->Predecessors.erase(SuccessorIt);
        Found = true;
        break;
      } else {
        ++SuccessorIt;
      }
    }

    revng_assert(Found);

    Base::Successors.erase(It.getCurrent());
  }

  void removeSuccessorEdge(edge_iterator It) {
    auto Edge = *It;
    bool Found = false;

    // Extract successor
    auto Successor = Edge.Neighbor;

    // Invert edge direction
    Edge.Neighbor = this;

    auto SuccessorPredecessors = Successor->predecessor_edges();
    for (auto SuccessorIt = SuccessorPredecessors.begin(),
              Last = SuccessorPredecessors.end();
         SuccessorIt != Last;) {
      if (Edge == *SuccessorIt) {
        Successor->Predecessors.erase(SuccessorIt);
        Found = true;
        break;
      } else {
        ++SuccessorIt;
      }
    }
    revng_assert(Found);

    this->Successors.erase(It);
  }

  void clearSuccessors() {
    while (Base::Successors.size())
      removeSuccessorEdge(Base::Successors.begin());
  }

public:
  llvm::iterator_range<const_child_iterator> predecessors() const {
    return this->toNeighborRange(Predecessors);
  }

  llvm::iterator_range<child_iterator> predecessors() {
    return this->toNeighborRange(Predecessors);
  }

  llvm::iterator_range<const_edge_iterator> predecessor_edges() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  llvm::iterator_range<edge_iterator> predecessor_edges() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  bool hasPredecessors() const { return Predecessors.size() != 0; }
  size_t predecessorCount() const { return Predecessors.size(); }
  bool hasPredecessor(BidirectionalNode const *P) const {
    auto Iterator = llvm::find_if(Predecessors,
                                  [P](auto &N) { return N.Neighbor == P; });
    return Iterator != Predecessors.end();
  }

private:
  NeighborContainer Predecessors;
};

namespace revng::detail {
/// The parameters deciding specifics of the base type `MutableEdgeNode`
/// extends are non-trivial. That's why those decision were wrapped
/// inside this struct to minimize clutter.
///
/// \note At this step `TheNode` has not been declared yet, thus we accept a
///       template parameter that has the same signature as `TheNode` that
///       will be declared later. This allows us to use it as if it was
///       declared, provided that only the real `TheNode` is used as this
///       argument.
template<typename Node,
         typename EdgeLabel,
         bool NeedsParent,
         size_t SmallSize,
         template<typename, typename, bool, size_t, typename, size_t, bool>
         class TheNode,
         typename FinalType,
         size_t ParentSmallSize,
         bool ParentHasEntryNode>
struct MutableEdgeNodeBaseTCalc {
  static constexpr bool
    NoDerivation = std::is_same_v<FinalType, std::false_type>;
  using NodeType = TheNode<Node,
                           EdgeLabel,
                           NeedsParent,
                           SmallSize,
                           FinalType,
                           ParentSmallSize,
                           ParentHasEntryNode>;
  using DerivedType = std::conditional_t<NoDerivation, NodeType, FinalType>;
  using GenericGraph = GenericGraph<DerivedType,
                                    ParentSmallSize,
                                    ParentHasEntryNode>;
  using ParentType = Parent<GenericGraph, Node>;
  using Result = std::conditional_t<NeedsParent, ParentType, Node>;
};

template<typename NodeType, typename LabelType>
struct OwningEdge {
  NodeType *Neighbor;
  std::unique_ptr<LabelType> Label;
};

template<typename NodeType, typename LabelType>
struct NonOwningEdge {
  NodeType *Neighbor;
  LabelType *Label;
};

template<typename NodeType, typename LabelType>
struct EdgeView {
  NodeType *Neighbor;
  LabelType *Label;

  explicit EdgeView(OwningEdge<NodeType, LabelType> &E) :
    Neighbor(E.Neighbor), Label(E.Label.get()) {}
  explicit EdgeView(NonOwningEdge<NodeType, LabelType> &E) :
    Neighbor(E.Neighbor), Label(E.Label) {}
};

template<typename NodeType, typename LabelType>
struct ConstEdgeView {
  NodeType const *Neighbor;
  LabelType const *Label;

  explicit ConstEdgeView(OwningEdge<NodeType, LabelType> const &E) :
    Neighbor(E.Neighbor), Label(E.Label.get()) {}
  explicit ConstEdgeView(NonOwningEdge<NodeType, LabelType> const &E) :
    Neighbor(E.Neighbor), Label(E.Label) {}
};
} // namespace revng::detail

/// A node type suitable for graphs where the edge labels are not cheap
/// to copy or need to be modified often.
template<typename Node,
         typename EdgeLabel,
         bool NeedsParent = true,
         size_t SmallSize = 2,
         typename FinalType = std::false_type,
         size_t ParentSmallSize = 16,
         bool ParentHasEntryNode = true>
class MutableEdgeNode
  : public revng::detail::MutableEdgeNodeBaseTCalc<Node,
                                                   EdgeLabel,
                                                   NeedsParent,
                                                   SmallSize,
                                                   MutableEdgeNode,
                                                   FinalType,
                                                   ParentSmallSize,
                                                   ParentHasEntryNode>::Result {
public:
  static constexpr bool is_mutable_edge_node = true;
  static constexpr bool HasParent = NeedsParent;
  using TypeCalc = revng::detail::MutableEdgeNodeBaseTCalc<Node,
                                                           EdgeLabel,
                                                           NeedsParent,
                                                           SmallSize,
                                                           MutableEdgeNode,
                                                           FinalType,
                                                           ParentSmallSize,
                                                           ParentHasEntryNode>;
  using DerivedType = typename TypeCalc::DerivedType;
  using Base = typename TypeCalc::Result;
  using NodeData = Node;
  using EdgeLabelData = EdgeLabel;

public:
  using Edge = EdgeLabel;
  using EdgeView = revng::detail::EdgeView<DerivedType, EdgeLabel>;
  using ConstEdgeView = revng::detail::ConstEdgeView<DerivedType, EdgeLabel>;

protected:
  using OwningEdge = revng::detail::OwningEdge<DerivedType, EdgeLabel>;
  using NonOwningEdge = revng::detail::NonOwningEdge<DerivedType, EdgeLabel>;
  using EdgeOwnerContainer = llvm::SmallVector<OwningEdge, SmallSize>;
  using EdgeViewContainer = llvm::SmallVector<NonOwningEdge, SmallSize>;

public:
  template<typename... ArgTypes>
  explicit MutableEdgeNode(ArgTypes &&...Args) :
    Base(std::forward<ArgTypes>(Args)...) {}

  MutableEdgeNode(const MutableEdgeNode &) = default;
  MutableEdgeNode(MutableEdgeNode &&) = default;
  MutableEdgeNode &operator=(const MutableEdgeNode &) = default;
  MutableEdgeNode &operator=(MutableEdgeNode &&) = default;

  NodeData &data() { return *this; }
  const NodeData &data() const { return *this; }

  NodeData copyData() const { return *this; }
  NodeData &&moveData() { return std::move(*this); }

public:
  // This stuff is needed by the DominatorTree implementation
  void printAsOperand(llvm::raw_ostream &, bool) const { revng_abort(); }

public:
  EdgeView addSuccessor(MutableEdgeNode *NewSuccessor, EdgeLabel EL = {}) {
    auto [Owner, View] = constructEdge(this, NewSuccessor, std::move(EL));
    auto &Output = Successors.emplace_back(std::move(Owner));
    NewSuccessor->Predecessors.emplace_back(std::move(View));
    return EdgeView(Output);
  }
  EdgeView addPredecessor(MutableEdgeNode *NewPredecessor, EdgeLabel EL = {}) {
    auto [Owner, View] = constructEdge(NewPredecessor, this, std::move(EL));
    auto &Output = NewPredecessor->Successors.emplace_back(std::move(Owner));
    Predecessors.emplace_back(std::move(View));
    return EdgeView(Output);
  }

protected:
  struct SuccessorFilters {
    static EdgeView toView(OwningEdge &E) { return EdgeView(E); }
    static ConstEdgeView toConstView(OwningEdge const &E) {
      return ConstEdgeView(E);
    }

    static DerivedType *&toNeighbor(OwningEdge &E) { return E.Neighbor; }
    static DerivedType const *const &toConstNeighbor(OwningEdge const &E) {
      return E.Neighbor;
    }
  };

  struct PredecessorFilters {
    static EdgeView toView(NonOwningEdge &E) { return EdgeView(E); }
    static ConstEdgeView toConstView(NonOwningEdge const &E) {
      return ConstEdgeView(E);
    }

    static DerivedType *&toNeighbor(NonOwningEdge &E) { return E.Neighbor; }
    static DerivedType const *const &toConstNeighbor(NonOwningEdge const &E) {
      return E.Neighbor;
    }
  };

private:
  template<typename IteratorType, typename FunctionType>
  using mapped = revng::mapped_iterator<IteratorType, FunctionType>;

  using SuccPointer = OwningEdge *;
  using CSuccPointer = const OwningEdge *;
  using PredPointer = NonOwningEdge *;
  using CPredPointer = const NonOwningEdge *;
  using RSuccPointer = std::reverse_iterator<OwningEdge *>;
  using CRSuccPointer = std::reverse_iterator<const OwningEdge *>;
  using RPredPointer = std::reverse_iterator<NonOwningEdge *>;
  using CRPredPointer = std::reverse_iterator<const NonOwningEdge *>;

  using SuccV = std::decay_t<decltype(SuccessorFilters::toView)>;
  using CSuccV = std::decay_t<decltype(SuccessorFilters::toConstView)>;
  using SuccN = std::decay_t<decltype(SuccessorFilters::toNeighbor)>;
  using CSuccN = std::decay_t<decltype(SuccessorFilters::toConstNeighbor)>;
  using PredV = std::decay_t<decltype(PredecessorFilters::toView)>;
  using CPredV = std::decay_t<decltype(PredecessorFilters::toConstView)>;
  using PredN = std::decay_t<decltype(PredecessorFilters::toNeighbor)>;
  using CPredN = std::decay_t<decltype(PredecessorFilters::toConstNeighbor)>;

public:
  using SuccessorEdgeIterator = mapped<SuccPointer, SuccV>;
  using ConstSuccessorEdgeIterator = mapped<CSuccPointer, CSuccV>;
  using SuccessorIterator = mapped<SuccPointer, SuccN>;
  using ConstSuccessorIterator = mapped<CSuccPointer, CSuccN>;

  using PredecessorEdgeIterator = mapped<PredPointer, PredV>;
  using ConstPredecessorEdgeIterator = mapped<CPredPointer, CPredV>;
  using PredecessorIterator = mapped<PredPointer, PredN>;
  using ConstPredecessorIterator = mapped<CPredPointer, CPredN>;

  using ReverseSuccessorEdgeIterator = mapped<RSuccPointer, SuccV>;
  using ConstReverseSuccessorEdgeIterator = mapped<CRSuccPointer, CSuccV>;
  using ReverseSuccessorIterator = mapped<RSuccPointer, SuccN>;
  using ConstReverseSuccessorIterator = mapped<CRSuccPointer, CSuccN>;

  using ReversePredecessorEdgeIterator = mapped<RPredPointer, PredV>;
  using ConstReversePredecessorEdgeIterator = mapped<CRPredPointer, CPredV>;
  using ReversePredecessorIterator = mapped<RPredPointer, PredN>;
  using ConstReversePredecessorIterator = mapped<CRPredPointer, CPredN>;

public:
  SuccessorEdgeIterator successor_edges_begin() {
    return revng::map_iterator(Successors.begin(), SuccessorFilters::toView);
  }
  ConstSuccessorEdgeIterator successor_edges_begin() const {
    return revng::map_iterator(Successors.begin(),
                               SuccessorFilters::toConstView);
  }
  ConstSuccessorEdgeIterator successor_edges_cbegin() const {
    return revng::map_iterator(Successors.cbegin(),
                               SuccessorFilters::toConstView);
  }
  ReverseSuccessorEdgeIterator successor_edges_rbegin() {
    return revng::map_iterator(Successors.rbegin(), SuccessorFilters::toView);
  }
  ConstReverseSuccessorEdgeIterator successor_edges_rbegin() const {
    return revng::map_iterator(Successors.rbegin(),
                               SuccessorFilters::toConstView);
  }
  ConstReverseSuccessorEdgeIterator successor_edges_crbegin() const {
    return revng::map_iterator(Successors.crbegin(),
                               SuccessorFilters::toConstView);
  }

  SuccessorEdgeIterator successor_edges_end() {
    return revng::map_iterator(Successors.end(), SuccessorFilters::toView);
  }
  ConstSuccessorEdgeIterator successor_edges_end() const {
    return revng::map_iterator(Successors.end(), SuccessorFilters::toConstView);
  }
  ConstSuccessorEdgeIterator successor_edges_cend() const {
    return revng::map_iterator(Successors.cend(),
                               SuccessorFilters::toConstView);
  }
  ReverseSuccessorEdgeIterator successor_edges_rend() {
    return revng::map_iterator(Successors.rend(), SuccessorFilters::toView);
  }
  ConstReverseSuccessorEdgeIterator successor_edges_rend() const {
    return revng::map_iterator(Successors.rend(),
                               SuccessorFilters::toConstView);
  }
  ConstReverseSuccessorEdgeIterator successor_edges_crend() const {
    return revng::map_iterator(Successors.crend(),
                               SuccessorFilters::toConstView);
  }

public:
  SuccessorIterator successors_begin() {
    return revng::map_iterator(Successors.begin(),
                               SuccessorFilters::toNeighbor);
  }
  ConstSuccessorIterator successors_begin() const {
    return revng::map_iterator(Successors.begin(),
                               SuccessorFilters::toConstNeighbor);
  }
  ConstSuccessorIterator successors_cbegin() const {
    return revng::map_iterator(Successors.cbegin(),
                               SuccessorFilters::toConstNeighbor);
  }
  ReverseSuccessorIterator successors_rbegin() {
    return revng::map_iterator(Successors.rbegin(),
                               SuccessorFilters::toNeighbor);
  }
  ConstReverseSuccessorIterator successors_rbegin() const {
    return revng::map_iterator(Successors.rbegin(),
                               SuccessorFilters::toConstNeighbor);
  }
  ConstReverseSuccessorIterator successors_crbegin() const {
    return revng::map_iterator(Successors.crbegin(),
                               SuccessorFilters::toConstNeighbor);
  }

  SuccessorIterator successors_end() {
    return revng::map_iterator(Successors.end(), SuccessorFilters::toNeighbor);
  }
  ConstSuccessorIterator successors_end() const {
    return revng::map_iterator(Successors.end(),
                               SuccessorFilters::toConstNeighbor);
  }
  ConstSuccessorIterator successors_cend() const {
    return revng::map_iterator(Successors.cend(),
                               SuccessorFilters::toConstNeighbor);
  }
  ReverseSuccessorIterator successors_rend() {
    return revng::map_iterator(Successors.rend(), SuccessorFilters::toNeighbor);
  }
  ConstReverseSuccessorIterator successors_rend() const {
    return revng::map_iterator(Successors.rend(),
                               SuccessorFilters::toConstNeighbor);
  }
  ConstReverseSuccessorIterator successors_crend() const {
    return revng::map_iterator(Successors.crend(),
                               SuccessorFilters::toConstNeighbor);
  }

public:
  PredecessorEdgeIterator predecessor_edges_begin() {
    return revng::map_iterator(Predecessors.begin(),
                               PredecessorFilters::toView);
  }
  ConstPredecessorEdgeIterator predecessor_edges_begin() const {
    return revng::map_iterator(Predecessors.begin(),
                               PredecessorFilters::toConstView);
  }
  ConstPredecessorEdgeIterator predecessor_edges_cbegin() const {
    return revng::map_iterator(Predecessors.cbegin(),
                               PredecessorFilters::toConstView);
  }
  ReversePredecessorEdgeIterator predecessor_edges_rbegin() {
    return revng::map_iterator(Predecessors.rbegin(),
                               PredecessorFilters::toView);
  }
  ConstReversePredecessorEdgeIterator predecessor_edges_rbegin() const {
    return revng::map_iterator(Predecessors.rbegin(),
                               PredecessorFilters::toConstView);
  }
  ConstReversePredecessorEdgeIterator predecessor_edges_crbegin() const {
    return revng::map_iterator(Predecessors.crbegin(),
                               PredecessorFilters::toConstView);
  }

  PredecessorEdgeIterator predecessor_edges_end() {
    return revng::map_iterator(Predecessors.end(), PredecessorFilters::toView);
  }
  ConstPredecessorEdgeIterator predecessor_edges_end() const {
    return revng::map_iterator(Predecessors.end(),
                               PredecessorFilters::toConstView);
  }
  ConstPredecessorEdgeIterator predecessor_edges_cend() const {
    return revng::map_iterator(Predecessors.cend(),
                               PredecessorFilters::toConstView);
  }
  ReversePredecessorEdgeIterator predecessor_edges_rend() {
    return revng::map_iterator(Predecessors.rend(), PredecessorFilters::toView);
  }
  ConstReversePredecessorEdgeIterator predecessor_edges_rend() const {
    return revng::map_iterator(Predecessors.rend(),
                               PredecessorFilters::toConstView);
  }
  ConstReversePredecessorEdgeIterator predecessor_edges_crend() const {
    return revng::map_iterator(Predecessors.crend(),
                               PredecessorFilters::toConstView);
  }

public:
  PredecessorIterator predecessors_begin() {
    return revng::map_iterator(Predecessors.begin(),
                               PredecessorFilters::toNeighbor);
  }
  ConstPredecessorIterator predecessors_begin() const {
    return revng::map_iterator(Predecessors.begin(),
                               PredecessorFilters::toConstNeighbor);
  }
  ConstPredecessorIterator predecessors_cbegin() const {
    return revng::map_iterator(Predecessors.cbegin(),
                               PredecessorFilters::toConstNeighbor);
  }
  ReversePredecessorIterator predecessors_rbegin() {
    return revng::map_iterator(Predecessors.rbegin(),
                               PredecessorFilters::toNeighbor);
  }
  ConstReversePredecessorIterator predecessors_rbegin() const {
    return revng::map_iterator(Predecessors.rbegin(),
                               PredecessorFilters::toConstNeighbor);
  }
  ConstReversePredecessorIterator predecessors_crbegin() const {
    return revng::map_iterator(Predecessors.crbegin(),
                               PredecessorFilters::toConstNeighbor);
  }

  PredecessorIterator predecessors_end() {
    return revng::map_iterator(Predecessors.end(),
                               PredecessorFilters::toNeighbor);
  }
  ConstPredecessorIterator predecessors_end() const {
    return revng::map_iterator(Predecessors.end(),
                               PredecessorFilters::toConstNeighbor);
  }
  ConstPredecessorIterator predecessors_cend() const {
    return revng::map_iterator(Predecessors.cend(),
                               PredecessorFilters::toConstNeighbor);
  }
  ReversePredecessorIterator predecessors_rend() {
    return revng::map_iterator(Predecessors.rend(),
                               PredecessorFilters::toNeighbor);
  }
  ConstReversePredecessorIterator predecessors_rend() const {
    return revng::map_iterator(Predecessors.rend(),
                               PredecessorFilters::toConstNeighbor);
  }
  ConstReversePredecessorIterator predecessors_crend() const {
    return revng::map_iterator(Predecessors.crend(),
                               PredecessorFilters::toConstNeighbor);
  }

public:
  llvm::iterator_range<SuccessorEdgeIterator> successor_edges() {
    return llvm::make_range(successor_edges_begin(), successor_edges_end());
  }
  llvm::iterator_range<ConstSuccessorEdgeIterator> successor_edges() const {
    return llvm::make_range(successor_edges_begin(), successor_edges_end());
  }
  llvm::iterator_range<SuccessorIterator> successors() {
    return llvm::make_range(successors_begin(), successors_end());
  }
  llvm::iterator_range<ConstSuccessorIterator> successors() const {
    return llvm::make_range(successors_begin(), successors_end());
  }

  llvm::iterator_range<PredecessorEdgeIterator> predecessor_edges() {
    return llvm::make_range(predecessor_edges_begin(), predecessor_edges_end());
  }
  llvm::iterator_range<ConstPredecessorEdgeIterator> predecessor_edges() const {
    return llvm::make_range(predecessor_edges_begin(), predecessor_edges_end());
  }
  llvm::iterator_range<PredecessorIterator> predecessors() {
    return llvm::make_range(predecessors_begin(), predecessors_end());
  }
  llvm::iterator_range<ConstPredecessorIterator> predecessors() const {
    return llvm::make_range(predecessors_begin(), predecessors_end());
  }

private:
  template<typename IteratorType>
  static auto findImpl(DerivedType const *N,
                       IteratorType FromIterator,
                       IteratorType ToIterator) {
    auto Comparator = [N](auto const &Edge) { return Edge.Neighbor == N; };
    return std::find_if(FromIterator, ToIterator, Comparator);
  }
  template<typename ContainerType>
  static auto findImpl(DerivedType const *N, ContainerType &&Where) {
    return findImpl(N, Where.begin(), Where.end());
  }

  static auto findSuccessorHalf(typename EdgeOwnerContainer::iterator Edge,
                                EdgeViewContainer &Halves) {
    auto Comparator = [Edge](auto const &Half) {
      return Half.Label == Edge->Label.get();
    };
    return std::find_if(Halves.begin(), Halves.end(), Comparator);
  }
  static auto findPredecessorHalf(typename EdgeViewContainer::iterator Edge,
                                  EdgeOwnerContainer &Halves) {
    auto Comparator = [Edge](auto const &Half) {
      return Half.Label.get() == Edge->Label;
    };
    return std::find_if(Halves.begin(), Halves.end(), Comparator);
  }

public:
  SuccessorEdgeIterator findSuccessorEdge(DerivedType const *S) {
    return SuccessorEdgeIterator(findImpl(S, Successors),
                                 SuccessorFilters::toView);
  }
  ConstSuccessorEdgeIterator findSuccessorEdge(DerivedType const *S) const {
    return ConstSuccessorEdgeIterator(findImpl(S, Successors),
                                      SuccessorFilters::toConstView);
  }
  PredecessorEdgeIterator findPredecessorEdge(DerivedType const *P) {
    return PredecessorEdgeIterator(findImpl(P, Predecessors),
                                   PredecessorFilters::toView);
  }
  ConstPredecessorEdgeIterator findPredecessorEdge(DerivedType const *P) const {
    return ConstPredecessorEdgeIterator(findImpl(P, Predecessors),
                                        PredecessorFilters::toConstView);
  }

  SuccessorIterator findSuccessor(DerivedType const *S) {
    return SuccessorIterator(findImpl(S, Successors),
                             SuccessorFilters::toNeighbor);
  }
  ConstSuccessorIterator findSuccessor(DerivedType const *S) const {
    return ConstSuccessorIterator(findImpl(S, Successors),
                                  SuccessorFilters::toConstNeighbor);
  }
  PredecessorIterator findPredecessor(DerivedType const *P) {
    return PredecessorIterator(findImpl(P, Predecessors),
                               PredecessorFilters::toNeighbor);
  }
  ConstPredecessorIterator findPredecessor(DerivedType const *P) const {
    return ConstPredecessorIterator(findImpl(P, Predecessors),
                                    PredecessorFilters::toConstNeighbor);
  }

public:
  bool hasSuccessor(DerivedType const *S) const {
    return findImpl(S, Successors) != Successors.end();
  }
  bool hasPredecessor(DerivedType const *P) const {
    return findImpl(P, Predecessors) != Predecessors.end();
  }

public:
  size_t successorCount() const { return Successors.size(); }
  size_t predecessorCount() const { return Predecessors.size(); }

  bool hasSuccessors() const { return Successors.size() != 0; }
  bool hasPredecessors() const { return Predecessors.size() != 0; }

protected:
  using InternalOwnerIt = typename EdgeOwnerContainer::const_iterator;
  using InternalViewIt = typename EdgeViewContainer::const_iterator;

  using InternalOwnerRIt = typename EdgeOwnerContainer::const_reverse_iterator;
  using InternalViewRIt = typename EdgeViewContainer::const_reverse_iterator;

protected:
  auto removeSuccessorImpl(InternalOwnerIt InputIterator) {
    if (Successors.empty())
      return Successors.end();

    auto Iterator = Successors.begin();
    std::advance(Iterator,
                 std::distance<InternalOwnerIt>(Iterator, InputIterator));
    revng_assert(Iterator != Successors.end());

    // Maybe we should do some extra checks as to whether `Iterator` is valid.
    auto *Successor = Iterator->Neighbor;
    revng_assert(!Successor->Predecessors.empty(),
                 "Half of an edge is missing, graph layout is broken.");

    auto PredecessorIt = findSuccessorHalf(Iterator, Successor->Predecessors);
    revng_assert(PredecessorIt != Successor->Predecessors.end(),
                 "Half of an edge is missing, graph layout is broken.");
    std::swap(*PredecessorIt, Successor->Predecessors.back());
    Successor->Predecessors.pop_back();

    auto AssertHelper = findSuccessorHalf(Iterator, Successor->Predecessors);
    revng_assert(AssertHelper == Successor->Predecessors.end(),
                 "More than one half is found for a single edge.");

    std::swap(*Iterator, Successors.back());
    Successors.pop_back();

    return Iterator;
  }
  auto removePredecessorImpl(InternalViewIt InputIterator) {
    if (Predecessors.empty())
      return Predecessors.end();

    auto Iterator = Predecessors.begin();
    std::advance(Iterator,
                 std::distance<InternalViewIt>(Iterator, InputIterator));
    revng_assert(Iterator != Predecessors.end());

    // Maybe we should do some extra checks as to whether `Iterator` is valid.
    auto *Predecessor = Iterator->Neighbor;
    revng_assert(!Predecessor->Successors.empty(),
                 "Half of an edge is missing, graph layout is broken.");

    auto SuccessorIt = findPredecessorHalf(Iterator, Predecessor->Successors);
    revng_assert(SuccessorIt != Predecessor->Successors.end(),
                 "Half of an edge is missing, graph layout is broken.");
    std::swap(*SuccessorIt, Predecessor->Successors.back());
    Predecessor->Successors.pop_back();

    auto AssertHelper = findPredecessorHalf(Iterator, Predecessor->Successors);
    revng_assert(AssertHelper == Predecessor->Successors.end(),
                 "More than one half is found for a single edge.");

    std::swap(*Iterator, Predecessors.back());
    Predecessors.pop_back();

    return Iterator;
  }

protected:
  auto removeSuccessorImpl(InternalOwnerRIt InputIterator) {
    auto Result = removeSuccessorImpl(std::prev(InputIterator.base()));
    return std::reverse_iterator(Result);
  }
  auto removePredecessorImpl(InternalViewRIt InputIterator) {
    auto Result = removePredecessorImpl(std::prev(InputIterator.base()));
    return std::reverse_iterator(Result);
  }

public:
  auto removeSuccessor(ConstSuccessorEdgeIterator Iterator) {
    auto Result = removeSuccessorImpl(Iterator.getCurrent());
    return SuccessorEdgeIterator(Result, SuccessorFilters::toView);
  }
  auto removeSuccessor(ConstSuccessorIterator Iterator) {
    auto Result = removeSuccessorImpl(Iterator.getCurrent());
    return SuccessorIterator(Result, SuccessorFilters::toNeighbor);
  }
  auto removeSuccessor(ConstReverseSuccessorEdgeIterator Iterator) {
    auto Result = removeSuccessorImpl(Iterator.getCurrent());
    return ReverseSuccessorEdgeIterator(Result, SuccessorFilters::toView);
  }
  auto removeSuccessor(ConstReverseSuccessorIterator Iterator) {
    auto Result = removeSuccessorImpl(Iterator.getCurrent());
    return ReverseSuccessorIterator(Result, SuccessorFilters::toNeighbor);
  }

  auto removePredecessor(ConstPredecessorEdgeIterator Iterator) {
    auto Result = removePredecessorImpl(Iterator.getCurrent());
    return ReverseSuccessorEdgeIterator(Result, PredecessorFilters::toView);
  }
  auto removePredecessor(ConstPredecessorIterator Iterator) {
    auto Result = removePredecessorImpl(Iterator.getCurrent());
    return ReverseSuccessorEdgeIterator(Result, PredecessorFilters::toNeighbor);
  }
  auto removePredecessor(ConstReversePredecessorEdgeIterator Iterator) {
    auto Result = removePredecessorImpl(Iterator.getCurrent());
    return ReversePredecessorEdgeIterator(Result, PredecessorFilters::toView);
  }
  auto removePredecessor(ConstReversePredecessorIterator Iterator) {
    auto Result = removePredecessorImpl(Iterator.getCurrent());
    return ReversePredecessorIterator(Result, PredecessorFilters::toNeighbor);
  }

public:
  auto removeSuccessor(SuccessorEdgeIterator Iterator) {
    auto Converted = ConstSuccessorEdgeIterator(Iterator.getCurrent(),
                                                SuccessorFilters::toConstView);
    return removeSuccessor(Converted);
  }
  auto removeSuccessor(SuccessorIterator Iterator) {
    auto Converted = ConstSuccessorIterator(Iterator.getCurrent(),
                                            SuccessorFilters::toConstNeighbor);
    return removeSuccessor(Converted);
  }
  auto removePredecessor(PredecessorEdgeIterator Iterator) {
    auto &F = PredecessorFilters::toConstView;
    auto Converted = ConstPredecessorEdgeIterator(Iterator.getCurrent(), F);
    return removePredecessor(Converted);
  }
  auto removePredecessor(PredecessorIterator Iterator) {
    auto Conv = ConstPredecessorIterator(Iterator.getCurrent(),
                                         PredecessorFilters::toConstNeighbor);
    return removePredecessor(Conv);
  }

public:
  auto removeSuccessor(ReverseSuccessorEdgeIterator Iterator) {
    auto &F = SuccessorFilters::toConstView;
    auto Conv = ConstReverseSuccessorEdgeIterator(Iterator.getCurrent(), F);
    return removeSuccessor(Conv);
  }
  auto removeSuccessor(ReverseSuccessorIterator Iterator) {
    auto &F = SuccessorFilters::toConstNeighbor;
    auto Converted = ConstReverseSuccessorIterator(Iterator.getCurrent(), F);
    return removeSuccessor(Converted);
  }
  auto removePredecessor(ReversePredecessorEdgeIterator Iterator) {
    auto &F = PredecessorFilters::toConstView;
    auto Conv = ConstReversePredecessorEdgeIterator(Iterator.getCurrent(), F);
    return removePredecessor(Conv);
  }
  auto removePredecessor(ReversePredecessorIterator Iterator) {
    auto &F = PredecessorFilters::toConstNeighbor;
    auto Converted = ConstReversePredecessorIterator(Iterator.getCurrent(), F);
    return removePredecessor(Converted);
  }

public:
  void removeSuccessors(DerivedType const *S) {
    auto Iterator = findImpl(S, Successors);
    while (Iterator != Successors.end()) {
      Iterator = removeSuccessorImpl(Iterator);
      Iterator = findImpl(S, Iterator, Successors.end());
    }
  }
  void removePredecessors(DerivedType const *P) {
    auto Iterator = findImpl(P, Predecessors);
    while (Iterator != Predecessors.end()) {
      Iterator = removePredecessorImpl(Iterator);
      Iterator = findImpl(P, Iterator, Predecessors.end());
    }
  }

public:
  void removeSuccessors() {
    for (auto It = Successors.begin(); It != Successors.end();)
      It = removeSuccessorImpl(It);
    revng_assert(Successors.empty());
  }
  void removePredecessors() {
    for (auto It = Predecessors.begin(); It != Predecessors.end();)
      It = removePredecessorImpl(It);
    revng_assert(Predecessors.empty());
  }

public:
  MutableEdgeNode &disconnect() {
    removeSuccessors();
    removePredecessors();
    return *this;
  }

protected:
  std::tuple<OwningEdge, NonOwningEdge>
  constructEdge(DerivedType *From, DerivedType *To, EdgeLabel &&EL) {
    OwningEdge O{ To, std::make_unique<EdgeLabel>(std::move(EL)) };
    NonOwningEdge V{ From, O.Label.get() };
    return { std::move(O), std::move(V) };
  }

private:
  EdgeOwnerContainer Successors;
  EdgeViewContainer Predecessors;
};

/// Simple data structure to hold the EntryNode of a GenericGraph
template<typename NodeT>
class EntryNode {
private:
  NodeT *EntryNode = nullptr;

public:
  NodeT *getEntryNode() const { return EntryNode; }
  void setEntryNode(NodeT *EntryNode) { this->EntryNode = EntryNode; }
};

/// Generic graph parametrized in the node type
///
/// This graph owns its nodes (but not the edges).
/// It can optionally have an elected entry point.
template<typename NodeT, size_t SmallSize, bool HasEntryNode>
class GenericGraph
  : public std::conditional_t<HasEntryNode, EntryNode<NodeT>, Empty> {
public:
  // NOLINTNEXTLINE
  static const bool is_generic_graph = true;
  using NodesContainer = llvm::SmallVector<std::unique_ptr<NodeT>, SmallSize>;
  using Node = NodeT;
  static constexpr bool hasEntryNode = HasEntryNode;

private:
  using nodes_iterator_impl = typename NodesContainer::iterator;
  using const_nodes_iterator_impl = typename NodesContainer::const_iterator;

public:
  GenericGraph() = default;
  GenericGraph(const GenericGraph &) = delete;
  GenericGraph(GenericGraph &&) = default;
  GenericGraph &operator=(const GenericGraph &) = delete;
  GenericGraph &operator=(GenericGraph &&) = default;

  bool verify() const debug_function {
    for (const std::unique_ptr<NodeT> &Node : Nodes)
      if (Node.get() == nullptr)
        return false;
    return true;
  }

public:
  static NodeT *getNode(std::unique_ptr<NodeT> &E) { return E.get(); }
  static const NodeT *getConstNode(const std::unique_ptr<NodeT> &E) {
    return E.get();
  }

  // TODO: these iterators will not work with llvm::filter_iterator,
  //       since the mapped type is not a reference
  using nodes_iterator = llvm::mapped_iterator<nodes_iterator_impl,
                                               decltype(&getNode)>;
  using const_nodes_iterator = llvm::mapped_iterator<const_nodes_iterator_impl,
                                                     decltype(&getConstNode)>;

  llvm::iterator_range<nodes_iterator> nodes() {
    return llvm::map_range(llvm::make_range(Nodes.begin(), Nodes.end()),
                           getNode);
  }

  llvm::iterator_range<const_nodes_iterator> nodes() const {
    return llvm::map_range(llvm::make_range(Nodes.begin(), Nodes.end()),
                           getConstNode);
  }

  size_t size() const { return Nodes.size(); }

public:
  nodes_iterator findNode(Node const *NodePtr) {
    auto Comparator = [&NodePtr](auto &N) { return N.get() == NodePtr; };
    auto InternalIt = std::find_if(Nodes.begin(), Nodes.end(), Comparator);
    return nodes_iterator(InternalIt, getNode);
  }
  const_nodes_iterator findNode(Node const *NodePtr) const {
    auto Comparator = [&NodePtr](auto &N) { return N.get() == NodePtr; };
    auto InternalIt = std::find_if(Nodes.begin(), Nodes.end(), Comparator);
    return nodes_iterator(InternalIt, getConstNode);
  }

public:
  bool hasNodes() const { return Nodes.size() != 0; }
  bool hasNode(Node const *NodePtr) const {
    return findNode(NodePtr) != Nodes.end();
  }

public:
  NodeT *addNode(std::unique_ptr<NodeT> &&Ptr) {
    Nodes.emplace_back(std::move(Ptr));
    if constexpr (NodeT::HasParent)
      Nodes.back()->setParent(this);
    return Nodes.back().get();
  }

  template<class... ArgTypes>
  NodeT *addNode(ArgTypes &&...A) {
    Nodes.push_back(std::make_unique<NodeT>(std::forward<ArgTypes>(A)...));
    if constexpr (NodeT::HasParent)
      Nodes.back()->setParent(this);
    return Nodes.back().get();
  }

  nodes_iterator removeNode(nodes_iterator It) {
    if constexpr (StrictSpecializationOfMutableEdgeNode<Node>)
      (*It.getCurrent())->disconnect();

    auto InternalIt = Nodes.erase(It.getCurrent());
    return nodes_iterator(InternalIt, getNode);
  }
  nodes_iterator removeNode(Node const *NodePtr) {
    return removeNode(findNode(NodePtr));
  }

public:
  nodes_iterator insertNode(nodes_iterator Where,
                            std::unique_ptr<NodeT> &&Ptr) {
    auto InternalIt = Nodes.insert(Where.getCurrent(), std::move(Ptr));
    return nodes_iterator(InternalIt, getNode);
  }
  template<class... ArgTypes>
  nodes_iterator insertNode(nodes_iterator Where, ArgTypes &&...A) {
    auto Pointer = std::make_unique<NodeT>(std::forward<ArgTypes>(A)...);
    auto InternalIt = Nodes.insert(Where.getCurrent(), std::move(Pointer));
    return nodes_iterator(InternalIt, getNode);
  }

public:
  void reserve(size_t Size) { Nodes.reserve(Size); }
  void clear() { Nodes.clear(); }

protected:
  NodesContainer Nodes;
};

//
// GraphTraits implementation for GenericGraph
//
namespace llvm {

/// Specializes GraphTraits<ForwardNode<...> *>
template<SpecializationOfForwardNode T>
struct GraphTraits<T *> {
public:
  using NodeRef = T *;
  using ChildIteratorType = std::conditional_t<std::is_const_v<T>,
                                               typename T::const_child_iterator,
                                               typename T::child_iterator>;

  using EdgeRef = typename T::Edge &;
  template<typename Ty, typename True, typename False>
  using if_const = std::conditional_t<std::is_const_v<Ty>, True, False>;
  using ChildEdgeIteratorType = if_const<T,
                                         typename T::const_edge_iterator,
                                         typename T::edge_iterator>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }

  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->successor_edges().begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->successor_edges().end();
  }

  static NodeRef edge_dest(EdgeRef Edge) { return Edge.Neighbor; }

  static NodeRef getEntryNode(NodeRef N) { return N; };
};

/// Specializes GraphTraits<llvm::Inverse<BidirectionalNode<...> *>>
template<StrictSpecializationOfBidirectionalNode T>
struct GraphTraits<llvm::Inverse<T *>> {
public:
  using NodeRef = T *;
  using ChildIteratorType = std::conditional_t<std::is_const_v<T>,
                                               typename T::const_child_iterator,
                                               typename T::child_iterator>;

public:
  static ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }

  static ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }

  static NodeRef getEntryNode(llvm::Inverse<NodeRef> N) { return N.Graph; };
};

// TODO: implement const version of GraphTraits
/// Specializes GraphTraits<MutableEdgeNode<...> *>
template<StrictSpecializationOfMutableEdgeNode T>
struct GraphTraits<T *> {
public:
  using NodeRef = T *;
  using EdgeRef = typename T::EdgeView;

private:
  using ChildNodeIt = decltype(std::declval<T>().successors().begin());
  using ChildEdgeIt = decltype(std::declval<T>().successor_edges().begin());

public:
  using ChildIteratorType = ChildNodeIt;
  using ChildEdgeIteratorType = ChildEdgeIt;

public:
  static auto child_begin(T *N) { return N->successors().begin(); }
  static auto child_end(T *N) { return N->successors().end(); }

  static auto child_edge_begin(T *N) { return N->successor_edges().begin(); }
  static auto child_edge_end(T *N) { return N->successor_edges().end(); }

  static T *edge_dest(EdgeRef Edge) { return Edge.Neighbor; }
  static T *getEntryNode(T *N) { return N; };
};

// TODO: implement const version of GraphTraits
/// Specializes GraphTraits<llvm::Inverse<MutableEdgeNode<...> *>>
template<StrictSpecializationOfMutableEdgeNode T>
struct GraphTraits<llvm::Inverse<T *>> {
public:
  using NodeRef = T *;
  using EdgeRef = typename T::EdgeView;

private:
  using ChildNodeIt = decltype(std::declval<T>().predecessors().begin());
  using ChildEdgeIt = decltype(std::declval<T>().predecessor_edges().begin());

public:
  using ChildIteratorType = ChildNodeIt;
  using ChildEdgeIteratorType = ChildEdgeIt;

public:
  static auto child_begin(T *N) { return N->predecessors().begin(); }
  static auto child_end(T *N) { return N->predecessors().end(); }

  static auto child_edge_begin(T *N) { return N->predecessor_edges().begin(); }
  static auto child_edge_end(T *N) { return N->predecessor_edges().end(); }

  static T *edge_dest(EdgeRef Edge) { return &Edge.Neighbor; }
  static T *getEntryNode(llvm::Inverse<T *> N) { return N.Graph; };
};

/// Specializes GraphTraits<GenericGraph<...> *>>
template<SpecializationOfGenericGraph T>
struct GraphTraits<T *>
  : public GraphTraits<std::conditional_t<std::is_const_v<T>,
                                          const typename T::Node *,
                                          typename T::Node *>> {

  using NodeRef = std::conditional_t<std::is_const_v<T>,
                                     const typename T::Node *,
                                     typename T::Node *>;
  using nodes_iterator = std::conditional_t<std::is_const_v<T>,
                                            typename T::const_nodes_iterator,
                                            typename T::nodes_iterator>;

  static NodeRef getEntryNode(T *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(T *G) { return G->nodes().begin(); }

  static nodes_iterator nodes_end(T *G) { return G->nodes().end(); }

  static size_t size(T *G) { return G->size(); }
};

/// Specializes GraphTraits<llvm::Inverse<GenericGraph<...> *>>>
template<SpecializationOfGenericGraph T>
struct GraphTraits<llvm::Inverse<T *>>
  : public GraphTraits<
      llvm::Inverse<std::conditional_t<std::is_const_v<T>,
                                       const typename T::Node *,
                                       typename T::Node *>>> {

  using NodeRef = std::conditional_t<std::is_const_v<T>,
                                     const typename T::Node *,
                                     typename T::Node *>;
  using nodes_iterator = std::conditional_t<std::is_const_v<T>,
                                            typename T::const_nodes_iterator,
                                            typename T::nodes_iterator>;

  static NodeRef getEntryNode(llvm::Inverse<T *> Inv) {
    // TODO: we might want to consider an option of having optional
    // `ExitNode`s as well, for consistency.
    return Inv.Graph->getEntryNode();
  }

  static nodes_iterator nodes_begin(llvm::Inverse<T *> Inv) {
    return Inv.Graph->nodes().begin();
  }

  static nodes_iterator nodes_end(llvm::Inverse<T *> Inv) {
    return Inv.Graph->nodes().end();
  }

  static size_t size(T *G) { return G->size(); }
};

} // namespace llvm
