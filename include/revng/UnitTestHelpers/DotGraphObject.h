#ifndef REVNG_DOTGRAPHOBJECT_H
#define REVNG_DOTGRAPHOBJECT_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <iosfwd>
#include <type_traits>
#include <vector>

// LLVM includes
#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

class DotNode {

  // Define the container for the successors and some useful helpers.
public:
  using child_container = std::vector<DotNode *>;
  using child_iterator = typename child_container::iterator;
  using child_const_iterator = typename child_container::const_iterator;
  using child_range = llvm::iterator_range<child_iterator>;
  using child_const_range = llvm::iterator_range<child_const_iterator>;

  using edge_container = std::vector<std::pair<DotNode *, DotNode *>>;
  using edge_iterator = typename edge_container::iterator;
  using edge_const_iterator = typename edge_container::const_iterator;
  using edge_range = llvm::iterator_range<edge_iterator>;
  using edge_const_range = llvm::iterator_range<edge_const_iterator>;

private:
  llvm::SmallString<8> Name;

  // Actual container for the pointers to the successors nodes.
  child_container Successors;
  edge_container SuccEdges;

public:
  DotNode(llvm::StringRef Name) : Name(Name) {}

public:
  child_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  child_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  edge_range edge_successors() {
    return llvm::make_range(SuccEdges.begin(), SuccEdges.end());
  }

  edge_const_range edge_successors() const {
    return llvm::make_range(SuccEdges.begin(), SuccEdges.end());
  }

  llvm::StringRef getName() const { return Name; }

  void addSuccessor(DotNode *Successor);
};

class DotGraph {
  static DotNode *ptrFromRef(std::unique_ptr<DotNode> &P) { return P.get(); }

  using PtrFromRefT = DotNode *(*) (std::unique_ptr<DotNode> &P);

  static const DotNode *constPtrFromRef(const std::unique_ptr<DotNode> &P) {
    return P.get();
  }

  using CPtrFromRefT = const DotNode *(*) (const std::unique_ptr<DotNode> &P);

public:
  using node_container = std::vector<std::unique_ptr<DotNode>>;

  using internal_iterator = typename node_container::iterator;
  using node_iterator = llvm::mapped_iterator<internal_iterator, PtrFromRefT>;
  using node_range = llvm::iterator_range<node_iterator>;

  using internal_const_iterator = typename node_container::const_iterator;
  using node_const_iterator = llvm::mapped_iterator<internal_const_iterator,
                                                    CPtrFromRefT>;
  using node_const_range = llvm::iterator_range<node_const_iterator>;

private:
  node_container Nodes;
  DotNode *EntryNode;

public:
  DotGraph() {}

public:
  /// \brief Parse a particularly well-formed GraphViz from a file.
  void
  parseDotFromFile(llvm::StringRef FileName, llvm::StringRef EntryName = "");

  node_range nodes() { return llvm::make_range(begin(), end()); }

  node_const_range nodes() const { return llvm::make_range(begin(), end()); }

  node_iterator begin() {
    return llvm::map_iterator(Nodes.begin(), ptrFromRef);
  }

  node_const_iterator begin() const {
    return llvm::map_iterator(Nodes.begin(), constPtrFromRef);
  }

  node_iterator end() { return llvm::map_iterator(Nodes.end(), ptrFromRef); }

  node_const_iterator end() const {
    return llvm::map_iterator(Nodes.end(), constPtrFromRef);
  }

  size_t size() const { return Nodes.size(); }

  DotNode *getEntryNode() const { return EntryNode; }

  DotNode *addNode(llvm::StringRef Name);

  DotNode *getNodeByName(llvm::StringRef Name);

private:
  /// \brief Actual implementation of the parser.
  void parseDotImpl(std::ifstream &F, llvm::StringRef EntryName);
};

namespace llvm {

template<>
struct GraphTraits<DotNode *> {
public:
  using NodeRef = DotNode *;
  using ChildIteratorType = DotNode::child_iterator;

  using EdgeRef = std::pair<NodeRef, NodeRef>;
  using ChildEdgeIteratorType = DotNode::edge_iterator;

protected:
  template<typename T>
  using unref_t = std::remove_reference_t<T>;

  using ChildT = unref_t<decltype(*std::declval<ChildIteratorType>())>;
  static_assert(std::is_same_v<NodeRef, ChildT>);

  using ChildEdgeT = unref_t<decltype(*std::declval<ChildEdgeIteratorType>())>;
  static_assert(std::is_same_v<EdgeRef, ChildEdgeT>);

public:
  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }

public:
  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return N->edge_successors().begin();
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return N->edge_successors().end();
  }

  static NodeRef edge_dest(EdgeRef E) { return E.second; };
};

template<>
struct GraphTraits<const DotNode *> {
  using NodeRef = const DotNode *;
  using EdgeRef = std::pair<const NodeRef, const NodeRef>;

protected:
  static const DotNode *toConstNode(const NodeRef P) { return P; }
  using ConstNode = const DotNode *(*) (const NodeRef);

  using const_node_it = llvm::mapped_iterator<DotNode::child_const_iterator,
                                              ConstNode>;

  static std::pair<const NodeRef, const NodeRef> toConstEdge(const EdgeRef E) {
    return E;
  }
  using ConstEdge = std::pair<const NodeRef, const NodeRef> (*)(const EdgeRef);

  using const_edge_it = llvm::mapped_iterator<DotNode::edge_const_iterator,
                                              ConstEdge>;

public:
  using ChildIteratorType = const_node_it;

  using ChildEdgeIteratorType = const_edge_it;

protected:
  template<typename T>
  using unref_t = std::remove_reference_t<T>;

  using ChildT = unref_t<decltype(*std::declval<ChildIteratorType>())>;
  static_assert(std::is_same_v<NodeRef, ChildT>);

  using ChildEdgeT = unref_t<decltype(*std::declval<ChildEdgeIteratorType>())>;
  static_assert(std::is_same_v<EdgeRef, ChildEdgeT>);

public:
  static inline ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(N->successors().begin(), toConstNode);
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(N->successors().end(), toConstNode);
  }

public:
  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    return llvm::map_iterator(N->edge_successors().begin(), toConstEdge);
  }

  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    return llvm::map_iterator(N->edge_successors().end(), toConstEdge);
  }

  static NodeRef edge_dest(EdgeRef E) { return E.second; };
};

template<>
struct GraphTraits<DotGraph *> : public GraphTraits<DotNode *> {
  using nodes_iterator = DotGraph::node_iterator;

  static NodeRef getEntryNode(DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(DotGraph *G) { return G->end(); }

  static size_t size(DotGraph *G) { return G->size(); }
};

template<>
struct GraphTraits<const DotGraph *> : public GraphTraits<const DotNode *> {
  using nodes_iterator = DotGraph::node_const_iterator;

  static NodeRef getEntryNode(const DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(const DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(const DotGraph *G) { return G->end(); }

  static size_t size(const DotGraph *G) { return G->size(); }
};

} // namespace llvm

#endif // REVNG_DOTGRAPHOBJECT_H
