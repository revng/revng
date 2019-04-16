#ifndef REVNGC_RESTRUCTURE_CFG_DOTGRAPHOBJECT_H
#define REVNGC_RESTRUCTURE_CFG_DOTGRAPHOBJECT_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <vector>

// LLVM includes
#include "llvm/ADT/GraphTraits.h"
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
  llvm::StringRef Name;

  // Actual container for the pointers to the successors nodes.
  links_container Successors;

public:
  DotNode(std::string &Name) : Name(Name) {}

public:
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  llvm::StringRef getName() {
    return Name;
  }

  void addSuccessor(DotNode *Successor);
};

inline DotNode *pointerFromReference(std::unique_ptr<DotNode> &P) {
  return P.get();
}

class DotGraph {

public:
  using links_container = std::vector<std::unique_ptr<DotNode>>;
  using links_underlying_iterator = typename links_container::iterator;
  using links_const_underlying_iterator =
    typename links_container::const_iterator;
  using links_iterator = TransformIterator<DotNode *,
                                           links_underlying_iterator>;
  using links_const_iterator =
    TransformIterator<DotNode *, links_const_underlying_iterator>;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

private:
  links_container Nodes;
  DotNode *EntryNode;

public:
  DotGraph() {}

public:

  /// \brief Parse a particularly well-formed GraphViz from any stream.
  template<typename StreamT>
  void parseDot(StreamT &);

  /// \brief Parse a particularly well-formed GraphViz from a file.
  void parseDotFromFile(std::string FileName);

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_iterator begin() {
    return links_iterator(Nodes.begin(), pointerFromReference);
  }

  links_iterator end() {
    return links_iterator(Nodes.end(), pointerFromReference);
  }

  size_t size() const { return Nodes.size(); }

  DotNode *getEntryNode() {
    return EntryNode;
  }

  DotNode *addNode(std::string &Name);
};

namespace llvm {

template<>
struct GraphTraits<DotGraph *> {
  using NodeRef = DotNode *;
  using ChildIteratorType = DotNode::links_iterator;
  using nodes_iterator = DotGraph::links_iterator;

  static NodeRef getEntryNode(DotGraph *G) { return G->getEntryNode(); }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }

  static nodes_iterator nodes_begin(DotGraph *G) { return G->begin(); }

  static nodes_iterator nodes_end(DotGraph *G) { return G->end(); }

  static size_t size(const DotGraph *G) { return G->size(); }
};

} // namespace llvm

#endif // REVNGC_RESTRUCTURE_CFG_DOTGRAPHOBJECT_H
