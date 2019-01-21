#ifndef BASICBLOCKNODE_H
#define BASICBLOCKNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <map>

// LLVM includes
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"

// Local libraries includes
#include "revng/Support/Debug.h"

// Forward declaration.
class CFG;

/// \brief Graph Node, representing a basic block
class BasicBlockNode {

public:
  using links_container = llvm::SmallVector<BasicBlockNode *, 2>;
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

  /// Flag to identify return basic blocks
  bool IsReturn;

  /// Name of the basic block.
  std::string Name;

  /// Pointer to the parent function.
  CFG *Parent;

  /// Flag to indicate if the node is a dummy node.
  bool Dummy;

  CFG *CollapsedRegion;

public:
  BasicBlockNode(llvm::BasicBlock *BB, CFG *Parent);

  BasicBlockNode(std::string Name, CFG *Parent, bool IsDummy = false);

public:
  bool isReturn();
  void setReturn();

  CFG *getParent();

  bool isDummy();

  // TODO: Check why this implementation is really necessary.
  void printAsOperand(llvm::raw_ostream &O, bool PrintType);

  void addSuccessor(BasicBlockNode *Successor);

  void removeSuccessor(BasicBlockNode *Successor);

  void addPredecessor(BasicBlockNode *Predecessor);

  void removePredecessor(BasicBlockNode *Predecessor);

  void updatePointers(std::map<BasicBlockNode *,
                      BasicBlockNode *> &SubstitutionMap);

  size_t successor_size() const { return Successors.size(); }
  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  BasicBlockNode *getPredecessorI(size_t i);

  BasicBlockNode *getSuccessorI(size_t i);

  size_t predecessor_size() const { return Predecessors.size(); }
  links_const_range predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }
  links_range predecessors() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  llvm::BasicBlock *basicBlock() const;
  void setBasicBlock(llvm::BasicBlock *NewBB);
  llvm::StringRef getName() const;
  std::string getNameStr() const;

  void setCollapsedCFG(CFG *Graph);

  bool isCollapsed();

  CFG *getCollapsedCFG();
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<> struct GraphTraits<BasicBlockNode *> {
  using NodeRef = BasicBlockNode *;
  using ChildIteratorType = BasicBlockNode::links_iterator;

  static NodeRef getEntryNode(BasicBlockNode *N) {
    return N;
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<> struct GraphTraits<Inverse<BasicBlockNode *>> {
  using NodeRef = BasicBlockNode *;
  using ChildIteratorType = BasicBlockNode::links_iterator;

  static NodeRef getEntryNode(Inverse<BasicBlockNode *> G) {
    return G.Graph;
  }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }
};

} // namespace llvm

#endif // BASICBLOCKNODE_H
