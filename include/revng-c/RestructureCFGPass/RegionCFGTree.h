#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <cstdlib>
#include <set>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/SmallMap.h"

#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFGPass/Utils.h"

template<class NodeT>
class MetaRegion;

template<typename NodeT>
inline bool InlineFilter(const typename llvm::GraphTraits<NodeT>::EdgeRef &E) {
  return !E.second.Inlined;
}

inline bool isASwitch(BasicBlockNode<llvm::BasicBlock *> *Node) {

  // TODO: remove this workaround for searching for switch nodes.
  if (Node->isCode() and Node->getOriginalNode()) {
    llvm::BasicBlock *OriginalBB = Node->getOriginalNode();
    llvm::Instruction *TerminatorBB = OriginalBB->getTerminator();
    return llvm::isa<llvm::SwitchInst>(TerminatorBB);
  }

  // The node may be an artifical node, therefore not an original switch.
  return false;
}

/// \brief The RegionCFG, a container for BasicBlockNodes
template<class NodeT = llvm::BasicBlock *>
class RegionCFG {

  using BBNodeT = BasicBlockNode<NodeT>;
  using BBNodeTUniquePtr = typename std::unique_ptr<BBNodeT>;
  using getPointerT = BBNodeT *(*) (BBNodeTUniquePtr &);
  using getConstPointerT = const BBNodeT *(*) (const BBNodeTUniquePtr &);

  static BBNodeT *getPointer(BBNodeTUniquePtr &Original) {
    return Original.get();
  }

  static_assert(std::is_same_v<decltype(&getPointer), getPointerT>);

  static const BBNodeT *getConstPointer(const BBNodeTUniquePtr &Original) {
    return Original.get();
  }

  static_assert(std::is_same_v<decltype(&getConstPointer), getConstPointerT>);

public:
  using DuplicationMap = std::map<llvm::BasicBlock *, size_t>;
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

  // Template type for the `NodePairFilteredGraph`. This must be template
  // because `llvm::DominatorTreeOnView` expects a template type as first
  // template parameter, so it must not be resolved beforehand.
  template<typename NodeRefT>
  using EFGT = EdgeFilteredGraph<NodeRefT, InlineFilter<NodeRefT>>;
  using FDomTree = llvm::DominatorTreeOnView<BasicBlockNodeT, false, EFGT>;
  using FPostDomTree = llvm::DominatorTreeOnView<BasicBlockNodeT, true, EFGT>;

private:
  /// Storage for basic block nodes, associated to their original counterpart
  ///
  links_container BlockNodes;

  /// Pointer to the entry basic block of this function
  BasicBlockNodeT *EntryNode;
  unsigned IDCounter = 0;
  std::string FunctionName;
  std::string RegionName;
  bool ToInflate = true;
  llvm::DominatorTreeBase<BasicBlockNodeT, false> DT;
  llvm::DominatorTreeBase<BasicBlockNodeT, true> PDT;

  FDomTree IFDT;
  FPostDomTree IFPDT;

private:
  template<typename GraphNodeT>
  void addSuccessorEdges(GraphNodeT N,
                         const std::map<GraphNodeT, BBNodeT *> &NodeMap) {
    BBNodeT *BBNode = NodeMap.at(N);
    using llvm::make_range;
    using GT = typename llvm::GraphTraits<GraphNodeT>;
    for (typename GT::NodeRef OriginalSucc :
         make_range(GT::child_begin(N), GT::child_end(N))) {
      BBNodeT *Successor = NodeMap.at(OriginalSucc);
      addPlainEdge<BBNodeT>({ BBNode, Successor });
    }
  }

  using CBBMap = const std::map<llvm::BasicBlock *, BBNodeT *>;

  template<>
  void
  addSuccessorEdges<llvm::BasicBlock *>(llvm::BasicBlock *BB, CBBMap &NodeMap) {

    BBNodeT *BBNode = NodeMap.at(BB);

    if (auto *Switch = dyn_cast<llvm::SwitchInst>(BB->getTerminator())) {

      revng_assert(BBNode->isCode() and isASwitch(BBNode));
      using EdgeInfo = typename BasicBlockNodeT::EdgeInfo;

      SmallMap<llvm::BasicBlock *, EdgeInfo, 16> LabeledEdges;

      for (llvm::SwitchInst::CaseHandle &Case : Switch->cases()) {
        llvm::BasicBlock *CaseBB = Case.getCaseSuccessor();
        llvm::ConstantInt *CaseValue = Case.getCaseValue();
        uint64_t LabelValue = CaseValue->getZExtValue();
        LabeledEdges[CaseBB].Labels.insert(LabelValue);
      }

      auto *DefaultBB = Switch->getDefaultDest();
      revng_assert(DefaultBB);
      LabeledEdges[DefaultBB].Labels.clear();

      for (const auto &[Succ, Label] : LabeledEdges)
        addEdge<BBNodeT>({ BBNode, NodeMap.at(Succ) }, Label);

    } else {

      using GT = llvm::GraphTraits<llvm::BasicBlock *>;
      for (llvm::BasicBlock *OriginalSuccBB :
           make_range(GT::child_begin(BB), GT::child_end(BB))) {
        BBNodeT *Successor = NodeMap.at(OriginalSuccBB);
        addPlainEdge<BBNodeT>({ BBNode, Successor });
      }
    }
  }

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

    // Create a new node for each node in Graph.
    using llvm::make_range;
    for (NodeRef N : make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      BBNodeT *BBNode = addNode(N);
      NodeToBBNodeMap[N] = BBNode;
    }

    // Do another iteration over all the nodes in the graph to create the edges
    // in the graph.
    for (NodeRef N : make_range(GT::nodes_begin(Graph), GT::nodes_end(Graph))) {
      addSuccessorEdges(N, NodeToBBNodeMap);
    }

    // Set the `EntryNode` BasicBlockNode reference.
    EntryNode = NodeToBBNodeMap.at(GT::getEntryNode(Graph));
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

  links_iterator end() {
    return llvm::map_iterator(BlockNodes.end(), getPointer);
  }

  links_const_iterator end() const {
    return llvm::map_iterator(BlockNodes.end(), getConstPointer);
  }

  size_t size() const { return BlockNodes.size(); }
  void setSize(size_t Size) { BlockNodes.reserve(Size); }

  BBNodeT *addNode(NodeT Node, llvm::StringRef Name);
  BBNodeT *addNode(NodeT Node) { return addNode(Node, Node->getName()); }

  BBNodeT *createCollapsedNode(RegionCFG *Collapsed) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Collapsed));
    return BlockNodes.back().get();
  }

  BBNodeT *addArtificialNode(llvm::StringRef Name = "dummy",
                             BasicBlockNodeType T = BasicBlockNodeType::Empty) {
    revng_assert(T == BasicBlockNodeType::Empty
                 or T == BasicBlockNodeType::Break
                 or T == BasicBlockNodeType::Continue);
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(this, Name, T));
    return BlockNodes.back().get();
  }

  BBNodeT *addContinue() {
    return addArtificialNode("continue", BasicBlockNodeT::Type::Continue);
  }

  BBNodeT *addBreak() {
    return addArtificialNode("break", BasicBlockNodeT::Type::Break);
  }

  BBNodeT *addDispatcher(llvm::StringRef Name) {
    using Type = typename BasicBlockNodeT::Type;
    using BBNodeT = BasicBlockNodeT;
    auto D = std::make_unique<BBNodeT>(this, Name, Type::Dispatcher);
    return BlockNodes.emplace_back(std::move(D)).get();
  }

  BBNodeT *addEntryDispatcher() { return addDispatcher("entry dispatcher"); }

  BBNodeT *addExitDispatcher() { return addDispatcher("exit dispatcher"); }

  BBNodeT *
  addSetStateNode(unsigned StateVariableValue, llvm::StringRef TargetName) {
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

  BBNodeT *addTile() {
    using Type = typename BasicBlockNodeT::Type;
    auto Tmp = std::make_unique<BBNodeT>(this, "tile", Type::Tile);
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

  std::vector<BBNodeTUniquePtr> &getNodes() { return BlockNodes; }

public:
  /// \brief Dump a GraphViz representing this function on any stream
  template<typename StreamT>
  void dumpDot(StreamT &) const;

  /// \brief Dump a GraphViz file on a file using an absolute path
  void dumpCFGOnFile(const std::string &FileName) const;

  /// \brief Dump a GraphViz file on a file using an absolute path
  void dumpCFGOnFile(const char *FName) const {
    return dumpCFGOnFile(std::string(FName));
  }

  /// \brief Dump a GraphViz file on a file representing this function
  void dumpCFGOnFile(const std::string &FunctionName,
                     const std::string &FolderName,
                     const std::string &FileName) const;

  bool purgeTrivialDummies();

  bool purgeIfTrivialDummy(BBNodeT *Dummy);

  void purgeVirtualSink(BBNodeT *Sink);

  BBNodeT *cloneUntilExit(BBNodeT *Node, BBNodeT *Sink);

  /// \brief Apply the untangle preprocessing pass.
  void untangle();

  /// \brief Apply comb to the region.
  void inflate();

  void removeNotReachables();

  void removeNotReachables(std::vector<MetaRegion<NodeT> *> &MS);

  bool isDAG();

  bool isTopologicallyEquivalent(RegionCFG &Other) const;

  void weave();

  void markUnexpectedAndAnyPCAsInlined();

protected:
  template<typename StreamT>
  void streamNode(StreamT &S, const BasicBlockNodeT *) const;
};

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
