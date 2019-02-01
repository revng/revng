#ifndef REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODE_H
#define REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODE_H

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

// revng includes
#include "revng/Support/Debug.h"

// forward declarations
class RegionCFG;

/// \brief Graph Node, representing a basic block
class BasicBlockNode {

public:

  enum class Type {
    Code,
    Empty,
    Break,
    Continue,
    Set,
    Check,
    Collapsed,
  };

  using links_container = llvm::SmallVector<BasicBlockNode *, 2>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

protected:
  /// Unique Node Id inside a RegionCFG, useful for printing to graphviz
  unsigned ID;

  /// Pointer to the parent RegionCFG
  RegionCFG *Parent;

  /// Reference to the corresponding basic block
  //
  // This is nullptr unless the BasicBlockNode represents a BasicBlock
  llvm::BasicBlock *BB;

  /// Reference to the corresponding collapsed region
  //
  // This is nullptr unless the BasicBlockNode represents a collapsed RegionCFG
  RegionCFG *CollapsedRegion;

  /// Flag to identify the exit type of a block
  Type NodeType;

  /// Name of the basic block.
  std::string Name;

  unsigned StateVariableValue;

  /// List of successors
  links_container Successors;

  /// List of predecessors
  links_container Predecessors;

  explicit BasicBlockNode(RegionCFG *Parent,
                          llvm::BasicBlock * BB,
                          RegionCFG *Collapsed,
                          const std::string &Name,
                          Type T,
                          unsigned Value = 0);

public:

  BasicBlockNode() = delete;
  BasicBlockNode(BasicBlockNode &&BBN) = delete;
  BasicBlockNode &operator=(const BasicBlockNode &BBN) = delete;
  BasicBlockNode &operator=(BasicBlockNode &&BBN) = delete;

  /// Copy ctor: clone the node in the same Parent with new ID and without edges
  explicit BasicBlockNode(const BasicBlockNode &BBN, RegionCFG *Parent) :
    BasicBlockNode(Parent,
                   BBN.BB,
                   BBN.CollapsedRegion,
                   BBN.Name,
                   BBN.NodeType,
                   BBN.StateVariableValue) {}

  /// \brief Constructor for nodes pointing to LLVM IR BasicBlock
  explicit BasicBlockNode(RegionCFG *Parent,
                          llvm::BasicBlock *BB,
                          const std::string &Name = "") :
    BasicBlockNode(Parent,
                   BB,
                   nullptr,
                   Name.size() ? Name : std::string(BB->getName()),
                   Type::Code) {}

  /// \brief Constructor for nodes representing collapsed subgraphs
  explicit BasicBlockNode(RegionCFG *Parent,
                          RegionCFG *Collapsed,
                          const std::string &Name = "") :
    BasicBlockNode(Parent,
                   nullptr,
                   Collapsed,
                   Name,
                   Type::Collapsed) {}

  /// \brief Constructor for empty dummy nodes
  explicit BasicBlockNode(RegionCFG *Parent,
                          const std::string &Name = "",
                          Type T = Type::Empty) :
    BasicBlockNode(Parent,
                   nullptr,
                   nullptr,
                   Name,
                   T) {
    revng_assert(T == Type::Empty or T == Type::Break or T == Type::Continue);
  }

  /// \brief Constructor for dummy nodes that handle the state variable
  explicit BasicBlockNode(RegionCFG *Parent,
                          Type T,
                          unsigned Value,
                          const std::string &Name = "") :
    BasicBlockNode(Parent,
                   nullptr,
                   nullptr,
                   Name,
                   T,
                   Value) {
    revng_assert(T == Type::Set or T == Type::Check);
  }

public:
  bool isBreak() const { return NodeType == Type::Break; }
  void setBreak() { NodeType = Type::Break; }

  bool isContinue() const { return NodeType == Type::Continue; }
  void setContinue() { NodeType = Type::Continue; }

  bool isSet() const { return NodeType == Type::Set; }
  void setSet() { NodeType = Type::Set; }

  bool isCheck() const { return NodeType == Type::Check; }
  void setCheck() { NodeType = Type::Check; }

  bool isEmpty() const { return NodeType == Type::Empty; }
  void setEmpty() { NodeType = Type::Empty; }

  RegionCFG *getParent() { return Parent; }
  void setParent(RegionCFG *P) { Parent = P; }

  bool isArtificial() const {
    return NodeType != Type::Code and NodeType != Type::Collapsed;
  }

  void removeNode();

  // TODO: Check why this implementation is really necessary.
  void printAsOperand(llvm::raw_ostream &O, bool PrintType);

  void addSuccessor(BasicBlockNode *Successor) {
    Successors.push_back(Successor);
  }

  void removeSuccessor(BasicBlockNode *Successor);

  void addPredecessor(BasicBlockNode *Predecessor) {
    Predecessors.push_back(Predecessor);
  }

  void removePredecessor(BasicBlockNode *Predecessor);

  void updatePointers(const std::map<BasicBlockNode *,
                      BasicBlockNode *> &SubstitutionMap);

  size_t successor_size() const { return Successors.size(); }
  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  BasicBlockNode *getPredecessorI(size_t i) { return Predecessors[i]; }
  BasicBlockNode *getSuccessorI(size_t i) { return Successors[i]; }

  size_t predecessor_size() const { return Predecessors.size(); }

  links_const_range predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  links_range predecessors() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  unsigned getID() const { return ID; }
  llvm::BasicBlock *isBasicBlock() const { return BB; }
  llvm::BasicBlock *getBasicBlock() { return BB; }

  llvm::StringRef getName() const { return Name; }
  std::string getNameStr() const { return Name; }
  void setName(const std::string &N) { Name = N; }

  bool isCollapsed() const { return NodeType == Type::Collapsed; }
  RegionCFG *getCollapsedCFG() { return CollapsedRegion; }
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

#endif //REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODE_H
