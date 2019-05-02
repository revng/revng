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
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng/Support/Transform.h"

template<class NodeT>
class MetaRegion;

/// \brief The RegionCFG, a container for BasicBlockNodes
template<class NodeT = llvm::BasicBlock *>
class RegionCFG {

public:
  using BasicBlockNodeT = typename BasicBlockNode<NodeT>::BasicBlockNodeT;
  using BasicBlockNodeType = typename BasicBlockNodeT::Type;
  using BasicBlockNodeTUP = typename std::unique_ptr<BasicBlockNode<NodeT>>;
  using BasicBlockNodeTSet = std::set<BasicBlockNodeT *>;
  using BasicBlockNodeTVect = std::vector<BasicBlockNodeT *>;
  using BasicBlockNodeTUPVect = std::vector<std::unique_ptr<BasicBlockNodeT>>;
  using BBNodeMap = typename BasicBlockNode<NodeT>::BBNodeMap;
  using RegionCFGT = typename BasicBlockNode<NodeT>::RegionCFGT;

  using EdgeDescriptor = typename BasicBlockNode<NodeT>::EdgeDescriptor;

  using links_container = std::vector<std::unique_ptr<BasicBlockNode<NodeT>>>;
  using internal_iterator = typename links_container::iterator;
  using internal_const_iterator = typename links_container::const_iterator;
  using links_iterator = TransformIterator<BasicBlockNode<NodeT> *,
                                           internal_iterator>;
  using links_const_iterator = TransformIterator<const BasicBlockNode<NodeT> *,
                                                 internal_const_iterator>;
  using links_range = llvm::iterator_range<links_iterator>;
  using links_const_range = llvm::iterator_range<links_const_iterator>;

  using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

private:
  /// Storage for basic block nodes, associated to their original counterpart
  ///
  links_container BlockNodes;

  /// Pointer to the entry basic block of this function
  BasicBlockNodeT *EntryNode;
  ASTTree AST;
  unsigned IDCounter = 0;
  std::string FunctionName;
  std::string RegionName;
  bool ToInflate = true;
  llvm::DominatorTreeBase<BasicBlockNodeT, false> DT;
  llvm::DominatorTreeBase<BasicBlockNodeT, true> PDT;

public:
  RegionCFG() = default;
  RegionCFG(const RegionCFG &) = default;
  RegionCFG(RegionCFG &&) = default;
  RegionCFG &operator=(const RegionCFG &) = default;
  RegionCFG &operator=(RegionCFG &&) = default;

  template<class GraphT>
  void initialize(GraphT Graph) {

    using GT = llvm::GraphTraits<GraphT>;
    using NodeRef = typename GT::NodeRef;

    // Map to keep the link between the original nodes and the BBNode created
    // from it.
    std::map<NodeRef, BasicBlockNode<NodeT> *> NodeToBBNodeMap;

    for (NodeRef N :
         llvm::make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      BasicBlockNode<NodeT> *BBNode = addNode(N);
      NodeToBBNodeMap[N] = BBNode;
    }

    // Set the `EntryNode` BasicBlockNode reference.
    EntryNode = NodeToBBNodeMap.at(GT::getEntryNode(Graph));

    // Do another iteration over all the nodes in the graph to create the edges
    // in the graph.
    for (NodeRef N :
         llvm::make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      BasicBlockNode<NodeT> *BBNode = NodeToBBNodeMap.at(N);

      // Iterate over all the successors of a graph node.
      unsigned ChildCounter = 0;
      for (NodeRef C : llvm::make_range(GT::child_begin(N), GT::child_end(N))) {

        // Check that no switches are present in the graph.
        revng_assert(ChildCounter < 2);
        ChildCounter++;

        // Create the edge in the RegionCFG<NodeT>.
        BasicBlockNode<NodeT> *Successor = NodeToBBNodeMap.at(C);
        BBNode->addSuccessor(Successor);
        Successor->addPredecessor(BBNode);
      }
    }
  }

  unsigned getNewID() { return IDCounter++; }

  links_range nodes() { return llvm::make_range(begin(), end()); }

  links_const_range nodes() const { return llvm::make_range(begin(), end()); }

  void setFunctionName(std::string Name);

  void setRegionName(std::string Name);

  std::string getFunctionName() const;

  std::string getRegionName() const;

  static inline BasicBlockNode<NodeT> *getPointer(BasicBlockNodeTUP &Original) {
    return Original.get();
  }

  static inline const BasicBlockNode<NodeT> *
  getConstPointer(const BasicBlockNodeTUP &Original) {
    return Original.get();
  }

  links_iterator begin() {
    return links_iterator(BlockNodes.begin(), getPointer);
  }

  links_const_iterator begin() const {
    return links_const_iterator(BlockNodes.begin(), getConstPointer);
  }

  links_iterator end() { return links_iterator(BlockNodes.end(), getPointer); }

  links_const_iterator end() const {
    return links_const_iterator(BlockNodes.end(), getConstPointer);
  }

  size_t size() const { return BlockNodes.size(); }
  void setSize(int Size) { BlockNodes.reserve(Size); }

  BasicBlockNode<NodeT> *addNode(NodeT Name);

  BasicBlockNode<NodeT> *createCollapsedNode(RegionCFG *Collapsed) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Collapsed));
    return BlockNodes.back().get();
  }

  BasicBlockNode<NodeT> *
  addArtificialNode(llvm::StringRef Name = "dummy",
                    BasicBlockNodeType T = BasicBlockNodeType::Empty) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Name, T));
    return BlockNodes.back().get();
  }

  BasicBlockNode<NodeT> *addContinue() {
    return addArtificialNode("continue", BasicBlockNodeT::Type::Continue);
  }
  BasicBlockNode<NodeT> *addBreak() {
    return addArtificialNode("break", BasicBlockNodeT::Type::Break);
  }

  BasicBlockNode<NodeT> *addDispatcher(unsigned StateVariableValue,
                                       BasicBlockNodeT *True,
                                       BasicBlockNodeT *False) {
    using Type = typename BasicBlockNodeT::Type;
    using BBNode = BasicBlockNodeT;
    std::string IdStr = std::to_string(StateVariableValue);
    // TODO: explicit in the check node the original destination nodes.
    std::string Name = "check idx " + IdStr;
    BlockNodes.emplace_back(std::make_unique<BBNode>(this,
                                                     Name,
                                                     Type::Check,
                                                     StateVariableValue));
    BBNode *Dispatcher = BlockNodes.back().get();
    Dispatcher->setTrue(True);
    Dispatcher->setFalse(False);
    return Dispatcher;
  }

  BasicBlockNode<NodeT> *
  addSetStateNode(unsigned StateVariableValue, llvm::StringRef TargetName) {
    using Type = typename BasicBlockNodeT::Type;
    using BBNode = BasicBlockNodeT;
    std::string IdStr = std::to_string(StateVariableValue);
    std::string Name = "set idx " + IdStr + " (desired target) "
                       + TargetName.str();
    BlockNodes.emplace_back(std::make_unique<BBNode>(this,
                                                     Name,
                                                     Type::Set,
                                                     StateVariableValue));
    return BlockNodes.back().get();
  }

  BasicBlockNode<NodeT> *cloneNode(BasicBlockNodeT &OriginalNode);

  void removeNode(BasicBlockNodeT *Node);

  void insertBulkNodes(BasicBlockNodeTSet &Nodes,
                       BasicBlockNodeT *Head,
                       BBNodeMap &SubstitutionMap);

  llvm::iterator_range<typename links_container::iterator>
  copyNodesAndEdgesFrom(RegionCFGT *O, BBNodeMap &SubstitutionMap);

  void connectBreakNode(std::set<EdgeDescriptor> &Outgoing,
                        const BBNodeMap &SubstitutionMap);

  void connectContinueNode();

  BasicBlockNode<NodeT> &getEntryNode() const { return *EntryNode; }

  BasicBlockNode<NodeT> &front() const { return *EntryNode; }

  std::vector<std::unique_ptr<BasicBlockNode<NodeT>>> &getNodes() {
    return BlockNodes;
  }

  std::vector<BasicBlockNode<NodeT> *>
  orderNodes(BasicBlockNodeTVect &List, bool DoReverse);

public:
  /// \brief Dump a GraphViz representing this function on any stream
  template<typename StreamT>
  void dumpDot(StreamT &) const;

  /// \brief Dump a GraphViz file on a file representing this function
  void dumpDotOnFile(std::string FolderName,
                     std::string FunctionName,
                     std::string FileName) const;

  /// \brief Dump a GraphViz file on a file using an absolute path
  void dumpDotOnFile(std::string FileName) const;

  std::vector<BasicBlockNode<NodeT> *> purgeDummies();

  void purgeVirtualSink(BasicBlockNode<NodeT> *Sink);

  std::vector<BasicBlockNode<NodeT> *>
  getInterestingNodes(BasicBlockNodeT *Condition);

  BasicBlockNode<NodeT> *cloneUntilExit(BasicBlockNode<NodeT> *Node,
                                        BasicBlockNode<NodeT> *Sink);

  /// \brief Apply the untangle preprocessing pass.
  void untangle();

  /// \brief Apply comb to the region.
  void inflate();

  void generateAst();

  // Get reference to the AST object which is inside the RegionCFG<NodeT> object
  ASTTree &getAST();

  void removeNotReachables();

  void removeNotReachables(std::vector<MetaRegion<NodeT> *> &MS);

  bool isDAG();

  bool isTopologicallyEquivalent(RegionCFG &Other) const;

protected:
  template<typename StreamT>
  void streamNode(StreamT &S, const BasicBlockNodeT *) const;
};

ASTNode *simplifyAtomicSequence(ASTNode *RootNode);

// Provide graph traits for usage with, e.g., llvm::ReversePostOrderTraversal
namespace llvm {

template<class NodeT>
struct GraphTraits<RegionCFG<NodeT> *>
  : public GraphTraits<BasicBlockNode<NodeT> *> {
  using nodes_iterator = typename RegionCFG<NodeT>::links_iterator;
  using NodeRef = BasicBlockNode<NodeT> *;

  static NodeRef getEntryNode(RegionCFG<NodeT> *F) {
    return &F->getEntryNode();
  }

  static nodes_iterator nodes_begin(RegionCFG<NodeT> *F) { return F->begin(); }

  static nodes_iterator nodes_end(RegionCFG<NodeT> *F) { return F->end(); }

  static size_t size(RegionCFG<NodeT> *F) { return F->size(); }
};

template<class NodeT>
struct GraphTraits<Inverse<RegionCFG<NodeT> *>>
  : public GraphTraits<Inverse<BasicBlockNode<NodeT> *>> {

  using NodeRef = BasicBlockNode<NodeT> *;

  static NodeRef getEntryNode(Inverse<RegionCFG<NodeT> *> G) {
    return &G.Graph->getEntryNode();
  }
};

} // namespace llvm

#endif // REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H
