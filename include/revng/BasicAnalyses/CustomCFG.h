#ifndef CUSTOMCFG_H
#define CUSTOMCFG_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <map>
#include <vector>

// Local libraries includes
#include "revng/Support/IRHelpers.h"

namespace llvm {
class BasicBlock;
}

/// \brief Node of a CustomCFG
///
/// A simple container for nodes and a set of successors and predecessors.
class CustomCFGNode {
public:
  using links_container = std::vector<CustomCFGNode *>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

private:
  /// List of successors
  links_container Successors;

  /// List of predecessors
  links_container Predecessors;

  /// Reference to the corresponding basic block
  llvm::BasicBlock *BB;

public:
  CustomCFGNode(llvm::BasicBlock *BB) : BB(BB) {}

  void addSuccessor(CustomCFGNode *Node) { Successors.push_back(Node); }

  void addPredecessor(CustomCFGNode *Node) { Predecessors.push_back(Node); }

  bool hasSuccessors() const { return Successors.size() != 0; }
  size_t successor_size() const { return Successors.size(); }
  links_const_range successors() const {
    return llvm::make_range(succ_begin(), succ_end());
  }
  links_range successors() {
    return llvm::make_range(succ_begin(), succ_end());
  }
  links_const_iterator succ_begin() const { return Successors.begin(); }
  links_const_iterator succ_end() const { return Successors.end(); }
  links_iterator succ_begin() { return Successors.begin(); }
  links_iterator succ_end() { return Successors.end(); }

  bool hasPredecessors() const { return Predecessors.size() != 0; }
  size_t predecessor_size() const { return Predecessors.size(); }
  links_const_range predecessors() const {
    return llvm::make_range(pred_begin(), pred_end());
  }
  links_const_iterator pred_begin() const { return Predecessors.begin(); }
  links_const_iterator pred_end() const { return Predecessors.end(); }
  links_range predecessors() {
    return llvm::make_range(pred_begin(), pred_end());
  }
  links_iterator pred_begin() { return Predecessors.begin(); }
  links_iterator pred_end() { return Predecessors.end(); }

  llvm::BasicBlock *block() const { return BB; }

  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Output, const char *Prefix) const {
    Output << Prefix << "Predecessors:\n";
    for (CustomCFGNode *Predecessor : predecessors())
      Output << Prefix << "  " << getName(Predecessor->block()) << "\n";
    Output << "\n";
    Output << Prefix << "Successors:\n";
    for (CustomCFGNode *Successor : successors())
      Output << Prefix << "  " << getName(Successor->block()) << "\n";
    Output << "\n";
  }
};

/// \brief A CFG representing a custom view on the actual CFG of a function
///
/// This class implements `GraphTraits`.
class CustomCFG {
public:
  void clear() { Blocks.clear(); }

  /// Populate the list of predecessors of each node based on the successors
  void buildBackLinks() {
    for (auto &P : Blocks)
      for (CustomCFGNode *Successor : P.second.successors())
        Successor->addPredecessor(&P.second);
  }

  bool hasNode(const llvm::BasicBlock *BB) const {
    return Blocks.count(BB) != 0;
  }

  CustomCFGNode *getNode(llvm::BasicBlock *BB) {
    auto It = Blocks.find(BB);
    if (It != Blocks.end())
      return &It->second;
    else
      return &Blocks.insert({ BB, CustomCFGNode(BB) }).first->second;
  }

  const CustomCFGNode *getNode(const llvm::BasicBlock *BB) const {
    return &Blocks.at(BB);
  }

  void dump() const debug_function { dump(dbg); }

  template<typename T>
  void dump(T &Output) const {
    for (auto &P : Blocks) {
      const llvm::BasicBlock *Block = P.first;
      const CustomCFGNode *Node = &P.second;
      Output << getName(Block) << ":\n";
      Node->dump(Output, "  ");
    }
  }

private:
  std::map<const llvm::BasicBlock *, CustomCFGNode> Blocks;
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<>
struct GraphTraits<CustomCFGNode *> {
  using NodeType = CustomCFGNode;
  using ChildIteratorType = CustomCFGNode::links_iterator;

  static NodeType *getEntryNode(CustomCFGNode *BB) { return BB; }

  static inline ChildIteratorType child_begin(CustomCFGNode *N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(CustomCFGNode *N) {
    return N->successors().end();
  }
};

} // namespace llvm

#endif // CUSTOMCFG_H
