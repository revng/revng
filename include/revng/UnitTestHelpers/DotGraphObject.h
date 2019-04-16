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
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

// Local libraries includes
#include "revng/Support/Transform.h"

class DotNode {

  // Define the container for the successors and some useful helpers.
public:
  using links_container = llvm::SmallVector<DotNode *, 2>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

private:
  llvm::SmallString<8> Name;

  // Actual container for the pointers to the successors nodes.
  links_container Successors;

public:
  DotNode(llvm::StringRef Name) : Name(Name) {}

public:
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  llvm::StringRef getName() const { return Name; }

  void addSuccessor(DotNode *Successor);
};

class DotGraph {

public:
  using links_container = std::vector<std::unique_ptr<DotNode>>;
  using internal_iterator = typename links_container::iterator;
  using internal_const_iterator = typename links_container::const_iterator;
  using links_iterator = TransformIterator<DotNode *, internal_iterator>;
  using links_const_iterator = TransformIterator<const DotNode *,
                                                 internal_const_iterator>;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

private:
  links_container Nodes;
  DotNode *EntryNode;

public:
  DotGraph() {}

public:
  /// \brief Parse a particularly well-formed GraphViz from a file.
  void
  parseDotFromFile(llvm::StringRef FileName, llvm::StringRef EntryName = "");

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_const_range nodes() const { return llvm::make_range(begin(), end()); }

  links_iterator begin() {
    return links_iterator(Nodes.begin(), pointerFromReference);
  }

  links_const_iterator begin() const {
    return links_const_iterator(Nodes.begin(), constPointerFromReference);
  }

  links_iterator end() {
    return links_iterator(Nodes.end(), pointerFromReference);
  }

  links_const_iterator end() const {
    return links_const_iterator(Nodes.end(), constPointerFromReference);
  }

  size_t size() const { return Nodes.size(); }

  DotNode *getEntryNode() const { return EntryNode; }

  DotNode *addNode(llvm::StringRef Name);

  DotNode *getNodeByName(llvm::StringRef Name);

private:
  static DotNode *pointerFromReference(std::unique_ptr<DotNode> &P) {
    return P.get();
  }

  static const DotNode *
  constPointerFromReference(const std::unique_ptr<DotNode> &P) {
    return P.get();
  }

  /// \brief Actual implementation of the parser.
  void parseDotImpl(std::ifstream &F, llvm::StringRef EntryName);
};

namespace llvm {

template<>
struct GraphTraits<DotNode *> {
  using NodeRef = DotNode *;
  using ChildIteratorType = DotNode::links_iterator;

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
  using ChildIteratorType = DotNode::links_const_iterator;

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<>
struct GraphTraits<DotGraph *> : public GraphTraits<DotNode *> {
  using nodes_iterator = DotGraph::links_iterator;

  static NodeRef getEntryNode(DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(DotGraph *G) { return G->end(); }

  static size_t size(DotGraph *G) { return G->size(); }
};

template<>
struct GraphTraits<const DotGraph *> : public GraphTraits<const DotNode *> {
  using nodes_iterator = DotGraph::links_const_iterator;

  static NodeRef getEntryNode(const DotGraph *G) { return G->getEntryNode(); }

  static nodes_iterator nodes_begin(const DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(const DotGraph *G) { return G->end(); }

  static size_t size(const DotGraph *G) { return G->size(); }
};

} // namespace llvm

#endif // REVNG_DOTGRAPHOBJECT_H
