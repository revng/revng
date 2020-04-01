#ifndef REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H
#define REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <llvm/ADT/STLExtras.h>
#include <set>

// LLVM includes
#include "llvm/IR/Dominators.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"

template<class NodeT>
class MetaRegion;

/// \brief The RegionCFG, a container for BasicBlockNodes
template<class NodeT = llvm::BasicBlock *>
class RegionCFG {

  using BBNodeT = BasicBlockNode<NodeT>;
  using BBNodeTUniquePtr = typename std::unique_ptr<BBNodeT>;
  using getPointerT = BBNodeT *(*)(BBNodeTUniquePtr &);
  using getConstPointerT = const BBNodeT *(*)(const BBNodeTUniquePtr &);

  static BBNodeT *getPointer(BBNodeTUniquePtr &Original) {
    return Original.get();
  }

  static_assert(std::is_same_v<decltype(&getPointer), getPointerT>);

  static const BBNodeT *
  getConstPointer(const BBNodeTUniquePtr &Original) {
    return Original.get();
  }

  static_assert(std::is_same_v<decltype(&getConstPointer), getConstPointerT>);

public:
  using BasicBlockNodeT = typename BBNodeT::BasicBlockNodeT;
  using BasicBlockNodeType = typename BasicBlockNodeT::Type;
  using BasicBlockNodeTSet = std::set<BasicBlockNodeT *>;
  using BasicBlockNodeTVect = std::vector<BasicBlockNodeT *>;
  using BasicBlockNodeTUPVect = std::vector<std::unique_ptr<BasicBlockNodeT>>;
  using BBNodeMap = typename BBNodeT::BBNodeMap;
  using RegionCFGT = typename BBNodeT::RegionCFGT;

  using EdgeDescriptor = typename BBNodeT::EdgeDescriptor;

  using links_container = std::vector<BBNodeTUniquePtr>;
  using internal_iterator = typename links_container::iterator;
  using internal_const_iterator = typename links_container::const_iterator;
  using links_iterator = llvm::mapped_iterator<internal_iterator, getPointerT>;
  using links_const_iterator = llvm::mapped_iterator<internal_const_iterator,
                                                     getConstPointerT>;
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
    std::map<NodeRef, BBNodeT *> NodeToBBNodeMap;

    for (NodeRef N :
         llvm::make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      BBNodeT *BBNode = addNode(N);
      NodeToBBNodeMap[N] = BBNode;
    }

    // Set the `EntryNode` BasicBlockNode reference.
    EntryNode = NodeToBBNodeMap.at(GT::getEntryNode(Graph));

    // Do another iteration over all the nodes in the graph to create the edges
    // in the graph.
    for (NodeRef N :
         llvm::make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      BBNodeT *BBNode = NodeToBBNodeMap.at(N);

      // Iterate over all the successors of a graph node.
      unsigned ChildCounter = 0;
      for (NodeRef C : llvm::make_range(GT::child_begin(N), GT::child_end(N))) {

        // Check that no switches are present in the graph.
        //revng_assert(ChildCounter < 2);
        ChildCounter++;

        // Create the edge in the RegionCFG<NodeT>.
        BasicBlockNode<NodeT> *Successor = NodeToBBNodeMap.at(C);

        // Check if the TerminatorInst of the node under analysis is a switch
        // instruction.
        /*
        llvm::Instruction *Terminator = BBNode->getTerminator();
        bool HasSwitch = false;
        if (Terminator->isTerminator()) {
          HasSwitch = true;
        }
        */

        // Do not connect the `unexpectedpc` successor in presence of a switch
        // node.
        // TODO: handle separately the default node in switches.
        if (!(Successor->getName() == "unexpectedpc")) {
          BBNode->addSuccessor(Successor);
          Successor->addPredecessor(BBNode);

          dbg << Successor->getName().str() << "\n";
        } else {
          BBNode->addSuccessor(Successor);
          Successor->addPredecessor(BBNode);

          dbg << Successor->getName().str() << "\n";
        }
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

  links_iterator begin() {
    return llvm::map_iterator(BlockNodes.begin(), getPointer);
  }

  links_const_iterator begin() const {
    return llvm::map_iterator(BlockNodes.begin(), getConstPointer);
  }

  links_iterator end() { return llvm::map_iterator(BlockNodes.end(), getPointer); }

  links_const_iterator end() const {
    return llvm::map_iterator(BlockNodes.end(), getConstPointer);
  }

  size_t size() const { return BlockNodes.size(); }
  void setSize(size_t Size) { BlockNodes.reserve(Size); }

  BBNodeT *addNode(NodeT Name);

  BBNodeT *createCollapsedNode(RegionCFG *Collapsed) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Collapsed));
    return BlockNodes.back().get();
  }

  BBNodeT *
  addArtificialNode(llvm::StringRef Name = "dummy",
                    BasicBlockNodeType T = BasicBlockNodeType::Empty) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Name, T));
    return BlockNodes.back().get();
  }

  BBNodeT *addContinue() {
    return addArtificialNode("continue", BasicBlockNodeT::Type::Continue);
  }
  BBNodeT *addBreak() {
    return addArtificialNode("break", BasicBlockNodeT::Type::Break);
  }

  BBNodeT *addDispatcher(unsigned StateVariableValue,
                                       BasicBlockNodeT *True,
                                       BasicBlockNodeT *False) {
    using Type = typename BasicBlockNodeT::Type;
    using BBNodeT = BasicBlockNodeT;
    std::string IdStr = std::to_string(StateVariableValue);
    // TODO: explicit in the check node the original destination nodes.
    std::string Name = "check idx " + IdStr;
    auto Tmp = std::make_unique<BBNodeT>(this,
                                         Name,
                                         Type::Check,
                                         StateVariableValue);
    BlockNodes.emplace_back(std::move(Tmp));
    BBNodeT *Dispatcher = BlockNodes.back().get();
    Dispatcher->setTrue(True);
    Dispatcher->setFalse(False);
    return Dispatcher;
  }

  BBNodeT *addDispatcherNew() {
    using Type = typename BasicBlockNodeT::Type;
    using Kind = typename BasicBlockNodeT::DispatcherKind;
    using BBNodeT = BasicBlockNodeT;
    std::string Name = "dispatcher node";
    auto Tmp = std::make_unique<BBNodeT>(this,
                                         Name,
                                         Type::Dispatcher,
                                         Kind::Entry);
    BlockNodes.emplace_back(std::move(Tmp));
    BBNodeT *Dispatcher = BlockNodes.back().get();
    return Dispatcher;
  }

  BBNodeT *addWeavingSwitch() {
    using Type = typename BasicBlockNodeT::Type;
    using Kind = typename BasicBlockNodeT::DispatcherKind;
    using BBNodeT = BasicBlockNodeT;
    std::string Name = "weaving switch";
    auto Tmp = std::make_unique<BBNodeT>(this,
                                         Name,
                                         Type::Dispatcher,
                                         Kind::Entry);
    BlockNodes.emplace_back(std::move(Tmp));
    BBNodeT *Dispatcher = BlockNodes.back().get();
    return Dispatcher;
  }

  BBNodeT *addSetStateNode(unsigned long StateVariableValue,
                           llvm::StringRef TargetName) {
    using Type = typename BasicBlockNodeT::Type;
    using BBNodeT = BasicBlockNodeT;
    std::string IdStr = std::to_string(StateVariableValue);
    std::string Name = "set idx " + IdStr + " (desired target) "
                       + TargetName.str();
    auto Tmp = std::make_unique<BBNodeT>(this,
                                         Name,
                                         Type::Set,
                                         StateVariableValue);
    BlockNodes.emplace_back(std::move(Tmp));
    return BlockNodes.back().get();
  }

  BBNodeT *cloneNode(BasicBlockNodeT &OriginalNode);

  void removeNode(BasicBlockNodeT *Node);

  void insertBulkNodes(BasicBlockNodeTSet &Nodes,
                       BasicBlockNodeT *Head,
                       BBNodeMap &SubstitutionMap);

  llvm::iterator_range<typename links_container::iterator>
  copyNodesAndEdgesFrom(RegionCFGT *O, BBNodeMap &SubstitutionMap);

  void connectBreakNode(std::set<EdgeDescriptor> &Outgoing,
                        const BBNodeMap &SubstitutionMap);

  void connectContinueNode();

  BBNodeT &getEntryNode() const { return *EntryNode; }

  BBNodeT &front() const { return *EntryNode; }

  std::vector<BBNodeTUniquePtr> &getNodes() {
    return BlockNodes;
  }

  std::vector<BBNodeT *>
  orderNodes(BasicBlockNodeTVect &List, bool DoReverse);

public:
  /// \brief Dump a GraphViz representing this function on any stream
  template<typename StreamT>
  void dumpDot(StreamT &) const;

  /// \brief Dump a GraphViz file on a file using an absolute path
  void dumpDotOnFile(const std::string &FileName) const;

  /// \brief Dump a GraphViz file on a file representing this function
  void dumpDotOnFile(const std::string &FolderName,
                     const std::string &FunctionName,
                     const std::string &FileName) const;

  std::vector<BBNodeT *> purgeDummies();

  void purgeVirtualSink(BBNodeT *Sink);

  std::vector<BBNodeT *>
  getInterestingNodes(BasicBlockNodeT *Condition);

  BBNodeT * cloneUntilExit(BBNodeT *Node, BBNodeT *Sink, bool AvoidSinking);

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

  void throttle();

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

extern unsigned DuplicationCounter;

extern unsigned UntangleTentativeCounter;
extern unsigned UntanglePerformedCounter;

#endif // REVNGC_RESTRUCTURE_CFG_REGIONCFGTREE_H
