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
                          llvm::BasicBlock *BB,
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
    BasicBlockNode(Parent, nullptr, Collapsed, Name, Type::Collapsed) {}

  /// \brief Constructor for empty dummy nodes
  explicit BasicBlockNode(RegionCFG *Parent,
                          const std::string &Name = "",
                          Type T = Type::Empty) :
    BasicBlockNode(Parent, nullptr, nullptr, Name, T) {
    revng_assert(T == Type::Empty or T == Type::Break or T == Type::Continue);
  }

  /// \brief Constructor for dummy nodes that handle the state variable
  explicit BasicBlockNode(RegionCFG *Parent,
                          Type T,
                          unsigned Value,
                          const std::string &Name = "") :
    BasicBlockNode(Parent, nullptr, nullptr, Name, T, Value) {
    revng_assert(T == Type::Set or T == Type::Check);
  }

public:
  bool isBreak() const { return NodeType == Type::Break; }
  bool isContinue() const { return NodeType == Type::Continue; }
  bool isSet() const { return NodeType == Type::Set; }
  bool isCheck() const { return NodeType == Type::Check; }
  bool isCode() const { return NodeType == Type::Code; }
  bool isEmpty() const { return NodeType == Type::Empty; }
  bool isArtificial() const {
    return NodeType != Type::Code and NodeType != Type::Collapsed;
  }
  Type getNodeType() const { return NodeType; }

  void setTrue(BasicBlockNode *Succ) {
    revng_assert(isCheck());
    Successors.resize(2, nullptr);
    if (Successors[1])
      Successors[1]->removePredecessor(this);
    Successors[1] = Succ;

    // We may not have a succesor (nullptr) or it may be already inserted
    // (insertBulkNodes on with check nodes around).
    // TODO: remove this check and handle this explicitly.
    if (Succ and (not Succ->hasPredecessor(this))) {
      Succ->addPredecessor(this);
    }
  }

  BasicBlockNode *getTrue() const {
    revng_assert(isCheck());
    revng_assert(successor_size() == 2);
    return Successors[1];
  }

  void setFalse(BasicBlockNode *Succ) {
    revng_assert(isCheck());
    Successors.resize(2, nullptr);
    if (Successors[0])
      Successors[0]->removePredecessor(this);
    Successors[0] = Succ;

    // We may not have a succesor (nullptr) or it may be already inserted
    // (insertBulkNodes on with check nodes around).
    // TODO: remove this check and handle this explicitly.
    if (Succ and (not Succ->hasPredecessor(this)))
      Succ->addPredecessor(this);
  }

  BasicBlockNode *getFalse() const {
    revng_assert(isCheck());
    revng_assert(successor_size() == 2);
    return Successors[0];
  }

  unsigned getStateVariableValue() const {
    revng_assert(isCheck() or isSet());
    return StateVariableValue;
  }

  RegionCFG *getParent() { return Parent; }
  void setParent(RegionCFG *P) { Parent = P; }

  void removeNode();

  // TODO: Check why this implementation is really necessary.
  void printAsOperand(llvm::raw_ostream &O, bool PrintType);

  void addSuccessor(BasicBlockNode *Successor) {
    // TODO: Disabled this, since even for set node if we copy the successors
    //       in order we should be fine.
    // revng_assert(not isCheck()); // you should use setFalse() and
    // setTrue()

    // Assert that we are not double inserting.
    bool Found = false;
    for (BasicBlockNode *Candidate : Successors) {
      if (Successor == Candidate) {
        Found = true;
        break;
      }
    }
    revng_assert(not Found);

    Successors.push_back(Successor);
  }

  void removeSuccessor(BasicBlockNode *Successor);

  void addPredecessor(BasicBlockNode *Predecessor) {

    // Assert that we are not double inserting.
    bool Found = false;
    for (BasicBlockNode *Candidate : Predecessors) {
      if (Predecessor == Candidate) {
        Found = true;
        break;
      }
    }
    revng_assert(not Found);

    Predecessors.push_back(Predecessor);
  }

  bool hasPredecessor(BasicBlockNode *Candidate) {

    // HACK to avoid double insertion due to `setFalse`, remove this.
    bool Found = false;
    for (BasicBlockNode *Predecessor : Predecessors) {
      if (Predecessor == Candidate) {
        Found = true;
        break;
      }
    }
    return Found;
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
  bool isBasicBlock() const { return NodeType == Type::Code; }
  llvm::BasicBlock *getBasicBlock() { return BB; }
  void setBasicBlock(llvm::BasicBlock *B) { BB = B; }

  llvm::StringRef getName() const { return Name; }
  std::string getNameStr() const {
    return "ID:" + std::to_string(getID()) + " " + Name;
  }
  void setName(const std::string &N) { Name = N; }

  bool isCollapsed() const { return NodeType == Type::Collapsed; }
  RegionCFG *getCollapsedCFG() { return CollapsedRegion; }
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<>
struct GraphTraits<BasicBlockNode *> {
  using NodeRef = BasicBlockNode *;
  using ChildIteratorType = BasicBlockNode::links_iterator;

  static NodeRef getEntryNode(BasicBlockNode *N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<>
struct GraphTraits<Inverse<BasicBlockNode *>> {
  using NodeRef = BasicBlockNode *;
  using ChildIteratorType = BasicBlockNode::links_iterator;

  static NodeRef getEntryNode(Inverse<BasicBlockNode *> G) { return G.Graph; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }
};

} // namespace llvm

#endif // REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODE_H
