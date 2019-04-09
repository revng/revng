#ifndef REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H
#define REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <set>

// LLVM includes
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

// Local libraries includes
#include "revng/Support/Transform.h"

// Local includes
#include "ASTTree.h"
#include "BasicBlockNode.h"

// Forward declaration
class MetaRegion;

inline BasicBlockNode *getPointer(std::unique_ptr<BasicBlockNode> &Original) {
  return Original.get();
}

/// \brief The RegionCFG, a container for BasicBlockNodes
class RegionCFG {

public:
  using links_container = std::vector<std::unique_ptr<BasicBlockNode>>;
  using links_underlying_iterator = typename links_container::iterator;
  using links_iterator = TransformIterator<BasicBlockNode *,
                                           links_underlying_iterator>;
  using links_range = llvm::iterator_range<links_iterator>;
  using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;
  using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

private:
  /// Storage for basic block nodes, associated to their original counterpart
  ///
  links_container BlockNodes;
  std::map<llvm::BasicBlock *, BasicBlockNode *> BBMap;

  /// Pointer to the entry basic block of this function
  llvm::BasicBlock *Entry;
  BasicBlockNode *EntryNode;
  ASTTree AST;
  unsigned IDCounter = 0;
  std::string FunctionName;
  std::string RegionName;
  bool ToInflate = true;
  llvm::DominatorTreeBase<BasicBlockNode, false> DT;
  llvm::DominatorTreeBase<BasicBlockNode, true> PDT;

public:
  RegionCFG() = default;
  RegionCFG(const RegionCFG &) = default;
  RegionCFG(RegionCFG &&) = default;
  RegionCFG &operator=(const RegionCFG &) = default;
  RegionCFG &operator=(RegionCFG &&) = default;

  void initialize(llvm::Function &F);

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() { return llvm::make_range(begin(), end()); }

  void setFunctionName(std::string Name);

  void setRegionName(std::string Name);

  std::string getFunctionName();

  std::string getRegionName();

  links_iterator begin() {
    return links_iterator(BlockNodes.begin(), getPointer);
  };

  links_iterator end() { return links_iterator(BlockNodes.end(), getPointer); };

  size_t size() const { return BlockNodes.size(); }
  void setSize(int Size) { BlockNodes.reserve(Size); }

  BasicBlockNode *addNode(llvm::BasicBlock *BB);

  BasicBlockNode *createCollapsedNode(RegionCFG *Collapsed,
                                      const std::string &Name = "collapsed") {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(this,
                                                             Collapsed,
                                                             Name));
    return BlockNodes.back().get();
  }

  BasicBlockNode *
  addArtificialNode(const std::string &Name = "",
                    BasicBlockNode::Type T = BasicBlockNode::Type::Empty) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(this, Name, T));
    return BlockNodes.back().get();
  }

  BasicBlockNode *addContinue(const std::string &Name = "continue") {
    return addArtificialNode(Name, BasicBlockNode::Type::Continue);
  }
  BasicBlockNode *addBreak(const std::string &Name = "break") {
    return addArtificialNode(Name, BasicBlockNode::Type::Break);
  }

  BasicBlockNode *addDispatcher(unsigned StateVariableValue,
                                BasicBlockNode *True,
                                BasicBlockNode *False) {
    using Type = BasicBlockNode::Type;
    using BBNode = BasicBlockNode;
    std::string IdStr = std::to_string(StateVariableValue);
    std::string NodeName = "check idx " + IdStr + " (true) "
                           + True->getNameStr() + " (false) "
                           + False->getNameStr();
    BlockNodes.emplace_back(std::make_unique<BBNode>(this,
                                                     Type::Check,
                                                     StateVariableValue,
                                                     NodeName));
    BBNode *Dispatcher = BlockNodes.back().get();
    Dispatcher->setTrue(True);
    Dispatcher->setFalse(False);
    return Dispatcher;
  }

  BasicBlockNode *
  addSetStateNode(unsigned StateVariableValue, const std::string &TargetName) {
    using Type = BasicBlockNode::Type;
    using BBNode = BasicBlockNode;
    std::string IdStr = std::to_string(StateVariableValue);
    std::string Name = "set idx " + IdStr + " (desired target) " + TargetName;
    BlockNodes.emplace_back(std::make_unique<BBNode>(this,
                                                     Type::Set,
                                                     StateVariableValue,
                                                     Name));
    return BlockNodes.back().get();
  }

  BasicBlockNode *cloneNode(const BasicBlockNode &OriginalNode);

  void removeNode(BasicBlockNode *Node);

  void insertBulkNodes(std::set<BasicBlockNode *> &Nodes,
                       BasicBlockNode *Head,
                       BBNodeMap &SubstitutionMap);

  llvm::iterator_range<links_container::iterator>
  copyNodesAndEdgesFrom(RegionCFG *O, BBNodeMap &SubstitutionMap);

  void connectBreakNode(std::set<std::pair<BasicBlockNode *,
                                           BasicBlockNode *>> &Outgoing,
                        BasicBlockNode *Break,
                        const BBNodeMap &SubstitutionMap);

  void connectContinueNode(BasicBlockNode *Continue);

  BasicBlockNode &get(llvm::BasicBlock *BB);

  BasicBlockNode &getEntryNode() { return *EntryNode; }

  BasicBlockNode &front() { return *EntryNode; }

  std::vector<std::unique_ptr<BasicBlockNode>> &getNodes() {
    return BlockNodes;
  }

  BasicBlockNode &getRandomNode();

  std::vector<BasicBlockNode *>
  orderNodes(std::vector<BasicBlockNode *> &List, bool DoReverse);

public:
  /// \brief Dump a GraphViz representing this function on any stream
  template<typename StreamT>
  void dumpDot(StreamT &) const;

  /// \brief Dump a GraphViz file on a file representing this function
  void dumpDotOnFile(std::string FolderName,
                     std::string FunctionName,
                     std::string FileName) const;

  void purgeDummies();

  void purgeVirtualSink(BasicBlockNode *Sink);

  std::vector<BasicBlockNode *> getInterestingNodes(BasicBlockNode *Condition);

  void inflate();

  void generateAst();

  // Get reference to the AST object which is inside the RegionCFG object
  ASTTree &getAST();

  void removeNotReachables();

  void removeNotReachables(std::vector<MetaRegion *> &MS);

  bool isDAG();

protected:
  template<typename StreamT>
  void streamNode(StreamT &S, const BasicBlockNode *) const;
};

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<>
struct GraphTraits<RegionCFG *> : public GraphTraits<BasicBlockNode *> {
  using nodes_iterator = RegionCFG::links_iterator;

  static NodeRef getEntryNode(RegionCFG *F) { return &F->getEntryNode(); }

  static nodes_iterator nodes_begin(RegionCFG *F) { return F->begin(); }

  static nodes_iterator nodes_end(RegionCFG *F) { return F->end(); }

  static size_t size(RegionCFG *F) { return F->size(); }
};

template<>
struct GraphTraits<Inverse<RegionCFG *>>
  : public GraphTraits<Inverse<BasicBlockNode *>> {

  static NodeRef getEntryNode(Inverse<RegionCFG *> G) {
    return &G.Graph->getEntryNode();
  }
};

} // namespace llvm

#endif // REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H
