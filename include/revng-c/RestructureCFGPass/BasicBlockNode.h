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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"

// revng includes
#include "revng/Support/Debug.h"

// forward declarations
template<class NodeT>
class RegionCFG;

/// \brief Graph Node, representing a basic block
template<class NodeT>
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

  using BasicBlockNodeT = BasicBlockNode<NodeT>;
  using BBNodeMap = std::map<BasicBlockNodeT *, BasicBlockNodeT *>;
  using RegionCFGT = RegionCFG<NodeT>;

  // EdgeDescriptor is a handy way to create and manipulate edges on the
  // RegionCFG.
  using EdgeDescriptor = std::pair<BasicBlockNodeT *, BasicBlockNodeT *>;

  using links_container = llvm::SmallVector<BasicBlockNodeT *, 2>;
  using links_iterator = typename links_container::iterator;
  using links_const_iterator = typename links_container::const_iterator;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

protected:
  /// Unique Node Id inside a RegionCFG<NodeT>, useful for printing to graphviz
  unsigned ID;

  /// Pointer to the parent RegionCFG<NodeT>
  RegionCFGT *Parent;

  /// Reference to the corresponding collapsed region
  //
  // This is nullptr unless the BasicBlockNode represents a collapsed
  // RegionCFG<NodeT>
  RegionCFGT *CollapsedRegion;

  /// Flag to identify the exit type of a block
  Type NodeType;

  /// Name of the basic block.
  llvm::SmallString<32> Name;

  unsigned StateVariableValue;

  /// List of successors
  links_container Successors;

  /// List of predecessors
  links_container Predecessors;

  // Original object pointer
  NodeT OriginalNode;

  explicit BasicBlockNode(RegionCFGT *Parent,
                          NodeT OriginalNode,
                          RegionCFGT *Collapsed,
                          llvm::StringRef Name,
                          Type T,
                          unsigned Value = 0);

public:
  BasicBlockNode() = delete;
  BasicBlockNode(BasicBlockNode &&BBN) = delete;
  BasicBlockNode &operator=(const BasicBlockNode &BBN) = delete;
  BasicBlockNode &operator=(BasicBlockNode &&BBN) = delete;

  /// Copy ctor: clone the node in the same Parent with new ID and without edges
  explicit BasicBlockNode(const BasicBlockNode &BBN, RegionCFGT *Parent) :
    BasicBlockNode(Parent,
                   BBN.OriginalNode,
                   BBN.CollapsedRegion,
                   BBN.Name,
                   BBN.NodeType,
                   BBN.StateVariableValue) {}

  /// \brief Constructor for nodes pointing to LLVM IR BasicBlock
  explicit BasicBlockNode(RegionCFGT *Parent,
                          NodeT OriginalNode,
                          llvm::StringRef Name = "") :
    BasicBlockNode(Parent, OriginalNode, nullptr, Name, Type::Code) {}

  /// \brief Constructor for nodes representing collapsed subgraphs
  explicit BasicBlockNode(RegionCFGT *Parent, RegionCFGT *Collapsed) :
    BasicBlockNode(Parent, nullptr, Collapsed, "collapsed", Type::Collapsed) {}

  /// \brief Constructor for empty dummy nodes
  explicit BasicBlockNode(RegionCFG<NodeT> *Parent,
                          llvm::StringRef Name,
                          Type T) :
    BasicBlockNode(Parent, nullptr, nullptr, Name, T) {
    revng_assert(T == Type::Empty or T == Type::Break or T == Type::Continue);
  }

  /// \brief Constructor for dummy nodes that handle the state variable
  explicit BasicBlockNode(RegionCFGT *Parent,
                          llvm::StringRef Name,
                          Type T,
                          unsigned Value) :
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
    if (Successors[1] and Successors[1]->hasPredecessor(this))
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
    if (Successors[0] and Successors[0]->hasPredecessor(this))
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

  RegionCFGT *getParent() { return Parent; }
  void setParent(RegionCFGT *P) { Parent = P; }

  void removeNode();

  // TODO: Check why this implementation is really necessary.
  void printAsOperand(llvm::raw_ostream &O, bool PrintType) const;

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

  void updatePointers(
    const std::map<BasicBlockNode *, BasicBlockNode *> &SubstitutionMap);

  size_t successor_size() const { return Successors.size(); }
  links_const_range successors() const {
    return llvm::make_range(Successors.begin(), Successors.end());
  }
  links_range successors() {
    return llvm::make_range(Successors.begin(), Successors.end());
  }

  BasicBlockNode *getPredecessorI(size_t i) const { return Predecessors[i]; }
  BasicBlockNode *getSuccessorI(size_t i) const { return Successors[i]; }

  size_t predecessor_size() const { return Predecessors.size(); }

  links_const_range predecessors() const {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  links_range predecessors() {
    return llvm::make_range(Predecessors.begin(), Predecessors.end());
  }

  unsigned getID() const { return ID; }
  bool isBasicBlock() const { return NodeType == Type::Code; }

  NodeT getOriginalNode() const { return OriginalNode; }

  llvm::StringRef getName() const;
  std::string getNameStr() const {
    return "ID:" + std::to_string(getID()) + " " + getName().str();
  }

  void setName(llvm::StringRef N) { Name = N; }

  bool isCollapsed() const { return NodeType == Type::Collapsed; }
  RegionCFGT *getCollapsedCFG() { return CollapsedRegion; }

  bool isEquivalentTo(BasicBlockNode *) const;
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<class NodeT>
struct GraphTraits<BasicBlockNode<NodeT> *> {
  using NodeRef = BasicBlockNode<NodeT> *;
  using ChildIteratorType = typename BasicBlockNode<NodeT>::links_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->successors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->successors().end();
  }
};

template<class NodeT>
struct GraphTraits<Inverse<BasicBlockNode<NodeT> *>> {
  using NodeRef = BasicBlockNode<NodeT> *;
  using ChildIteratorType = typename BasicBlockNode<NodeT>::links_iterator;

  static NodeRef getEntryNode(Inverse<NodeRef> G) { return G.Graph; }

  static inline ChildIteratorType child_begin(NodeRef N) {
    return N->predecessors().begin();
  }

  static inline ChildIteratorType child_end(NodeRef N) {
    return N->predecessors().end();
  }
};

} // namespace llvm

#endif // REVNGC_RESTRUCTURE_CFG_BASICBLOCKNODE_H
