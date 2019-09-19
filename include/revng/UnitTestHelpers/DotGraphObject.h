#ifndef REVNG_DOTGRAPHOBJECT_H
#define REVNG_DOTGRAPHOBJECT_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <iosfwd>
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
  using child_container = llvm::SmallVector<DotNode *, 2>;
  using child_iterator = typename child_container::iterator;
  using child_const_iterator = typename child_container::const_iterator;
  using child_range = llvm::iterator_range<child_iterator>;
  using child_const_range = llvm::iterator_range<child_const_iterator>;

private:
  llvm::SmallString<8> Name;

  // Actual container for the pointers to the successors nodes.
  child_container Successors;

public:
  DotNode(llvm::StringRef Name) : Name(Name) {}

public:
  child_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  child_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  llvm::StringRef getName() const { return Name; }

  void addSuccessor(DotNode *Successor);
};

class DotGraph {
  static DotNode *ptrFromRef(std::unique_ptr<DotNode> &P) { return P.get(); }

  static const DotNode *constPtrFromRef(const std::unique_ptr<DotNode> &P) {
    return P.get();
  }

  using PtrFromRefT = DotNode *(*) (std::unique_ptr<DotNode> &P);
  using CPtrFromRefT = const DotNode *(*) (const std::unique_ptr<DotNode> &P);

public:
  using child_container = std::vector<std::unique_ptr<DotNode>>;
  using internal_iterator = typename child_container::iterator;
  using internal_const_iterator = typename child_container::const_iterator;
  using child_iterator = llvm::mapped_iterator<internal_iterator, PtrFromRefT>;
  using child_const_iterator = llvm::mapped_iterator<internal_const_iterator,
                                                     CPtrFromRefT>;
  using child_range = llvm::iterator_range<child_iterator>;
  using child_const_range = llvm::iterator_range<child_const_iterator>;

private:
  child_container Nodes;
  DotNode *EntryNode;

public:
  DotGraph() {}

public:
  /// \brief Parse a particularly well-formed GraphViz from a file.
  void
  parseDotFromFile(llvm::StringRef FileName, llvm::StringRef EntryName = "");

  child_range nodes() { return llvm::make_range(begin(), end()); }

  child_const_range nodes() const { return llvm::make_range(begin(), end()); }

  child_iterator begin() {
    return llvm::map_iterator(Nodes.begin(), ptrFromRef);
  }

  child_const_iterator begin() const {
    return llvm::map_iterator(Nodes.begin(), constPtrFromRef);
  }

  child_iterator end() { return llvm::map_iterator(Nodes.begin(), ptrFromRef); }

  child_const_iterator end() const {
    return llvm::map_iterator(Nodes.begin(), constPtrFromRef);
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
  using NodeRef = DotNode *;
  using ChildIteratorType = DotNode::child_iterator;

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<>
struct GraphTraits<const DotNode *> {
  using NodeRef = const DotNode *;
  using ChildIteratorType = DotNode::child_const_iterator;

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<>
struct GraphTraits<DotGraph *> : public GraphTraits<DotNode *> {
  using nodes_iterator = DotGraph::child_iterator;

  static NodeRef getEntryNode(DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(DotGraph *G) { return G->end(); }

  static size_t size(DotGraph *G) { return G->size(); }
};

template<>
struct GraphTraits<const DotGraph *> : public GraphTraits<const DotNode *> {
  using nodes_iterator = DotGraph::child_const_iterator;

  static NodeRef getEntryNode(const DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(const DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(const DotGraph *G) { return G->end(); }

  static size_t size(const DotGraph *G) { return G->size(); }
};

} // namespace llvm

#endif // REVNG_DOTGRAPHOBJECT_H
