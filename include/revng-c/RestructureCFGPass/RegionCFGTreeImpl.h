#ifndef REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEIMPL_H
#define REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEIMPL_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/stat.h>

// LLVM includes
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"

// revng includes
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/ADT/ReversePostOrderTraversal.h"
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFGPass/MetaRegionBB.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

template<typename IterT>
bool intersect(IterT I1, IterT E1, IterT I2, IterT E2) {
  while ((I1 != E1) and (I2 != E2)) {
    if (*I1 < *I2)
      ++I1;
    else if (*I2 < *I1)
      ++I2;
    else
      return true;
  }
  return false;
}

template<typename IterT>
bool disjoint(IterT I1, IterT E1, IterT I2, IterT E2) {
  return not intersect(I1, E1, I2, E2);
}

template<typename RangeT>
bool intersect(const RangeT &R1, const RangeT &R2) {
  return intersect(R1.begin(), R1.end(), R2.begin(), R2.end());
}

template<typename RangeT>
bool disjoint(const RangeT &R1, const RangeT &R2) {
  return not intersect(R1, R2);
}

unsigned const SmallSetSize = 16;

// llvm::SmallPtrSet is a handy way to store set of BasicBlockNode pointers.
template<class NodeT>
using SmallPtrSet = llvm::SmallPtrSet<BasicBlockNode<NodeT> *, SmallSetSize>;

// Helper function that visit an AST tree and creates the sequence nodes
inline ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
  SequenceNode *RootSequenceNode = Tree.addSequenceNode();
  RootSequenceNode->addNode(RootNode);

  for (ASTNode *Node : RootSequenceNode->nodes()) {
    if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
      if (If->hasThen()) {
        If->setThen(createSequence(Tree, If->getThen()));
      }
      if (If->hasElse()) {
        If->setElse(createSequence(Tree, If->getElse()));
      }
    } else if (llvm::isa<CodeNode>(Node)) {
      // TODO: confirm that doesn't make sense to process a code node.
    } else if (llvm::isa<ScsNode>(Node)) {
      // TODO: confirm that this phase is not needed since the processing is
      //       done inside the processing of each SCS region.
    } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(Node)) {
      for (size_t I = 0; I < Switch->CaseSize(); I++) {
        Switch->replaceCaseN(I, createSequence(Tree, Switch->getCaseN(I)));
      }
      if (ASTNode *Default = Switch->getDefault()) {
        Switch->replaceDefault(createSequence(Tree, Default));
      }
    } else if (llvm::isa<BreakNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<ContinueNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<SequenceNode>(Node)) {
      // Stop here during the analysis.
    } else if (llvm::isa<SetNode>(Node)) {
      // Stop here during the analysis.
    } else {
      revng_abort("AST node type not expected");
    }
  }

  return RootSequenceNode;
}

// Helper function that simplifies useless dummy nodes
inline void simplifyDummies(ASTNode *RootNode) {

  switch (RootNode->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(RootNode);
    std::vector<ASTNode *> UselessDummies;
    for (ASTNode *Node : Sequence->nodes()) {
      if (Node->isEmpty()) {
        UselessDummies.push_back(Node);
      } else {
        simplifyDummies(Node);
      }
    }
    for (ASTNode *Node : UselessDummies) {
      Sequence->removeNode(Node);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(RootNode);
    if (If->hasThen()) {
      simplifyDummies(If->getThen());
    }
    if (If->hasElse()) {
      simplifyDummies(If->getElse());
    }
  } break;

  case ASTNode::NK_SwitchRegular:
  case ASTNode::NK_SwitchDispatcher: {

    auto *Switch = llvm::cast<SwitchNode>(RootNode);

    for (ASTNode *CaseNode : Switch->unordered_cases())
      simplifyDummies(CaseNode);

    if (auto *Default = Switch->getDefault())
      simplifyDummies(Default);

  } break;

  case ASTNode::NK_Scs:
  case ASTNode::NK_Code:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Set:
    // Do nothing
    break;

  default:
    revng_unreachable();
  }
}

// Helper function which simplifies sequence nodes composed by a single AST
// node.
inline ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {
  switch (RootNode->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(RootNode);
    switch (Sequence->listSize()) {

    case 0:
      RootNode = nullptr;
      break;

    case 1:
      RootNode = simplifyAtomicSequence(Sequence->getNodeN(0));
      break;

    default:
      bool Empty = true;
      for (ASTNode *&Node : Sequence->nodes()) {
        Node = simplifyAtomicSequence(Node);
        if (nullptr != Node)
          Empty = false;
      }
      revng_assert(not Empty);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(RootNode);

    if (If->hasThen())
      If->setThen(simplifyAtomicSequence(If->getThen()));

    if (If->hasElse())
      If->setElse(simplifyAtomicSequence(If->getElse()));

  } break;

  case ASTNode::NK_SwitchRegular:
  case ASTNode::NK_SwitchDispatcher: {

    auto *Switch = llvm::cast<SwitchNode>(RootNode);

    // Simplify any possible case node which is constituted only by a sequence
    // of dummy nodes.
    for (unsigned I = 0; I < Switch->CaseSize(); I++) {
      auto *CaseNode = Switch->getCaseN(I);
      auto *NewCaseNode = simplifyAtomicSequence(CaseNode);

      // If the recursive call of `simplifyAtomicSequence` returned a `nullptr`
      // (meaning we simplified the body to an empty sequence), remove the case
      // from the switch.
      if (NewCaseNode == nullptr) {

        // Invoke the removal case method of the parent `SwitchNode` class. This
        // method will take care of also invoking the method on the two
        // subclasses.
        Switch->removeCaseN(I);
      } else if (NewCaseNode != CaseNode) {
        Switch->replaceCaseN(I, NewCaseNode);
      }
    }

    // In case the recursive call to `simplifyAtomicSequence` gives origin to a
    // complete simplification of the default node of the switch, setting its
    // corresponding `ASTNode` to `nullptr` already does the job, since having
    // the corresponding `Default` field set to `nullptr` means that the switch
    // node has no default.
    if (ASTNode *Default = Switch->getDefault()) {
      auto *NewDefault = simplifyAtomicSequence(Default);
      if (NewDefault != Default)
        Switch->replaceDefault(NewDefault);
    }

  } break;

  case ASTNode::NK_Scs: {
    // TODO: check if this is not needed as the simplification is done for each
    //       SCS region.
    // After flattening this situation may arise again.
    auto *Scs = llvm::cast<ScsNode>(RootNode);
    if (Scs->hasBody())
      Scs->setBody(simplifyAtomicSequence(Scs->getBody()));
  } break;

  case ASTNode::NK_Code:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Set:
    // Do nothing
    break;

  default:
    revng_unreachable();
  }

  return RootNode;
}

template<class NodeT>
inline void RegionCFG<NodeT>::setFunctionName(std::string Name) {
  FunctionName = Name;
}

template<class NodeT>
inline void RegionCFG<NodeT>::setRegionName(std::string Name) {
  RegionName = Name;
}

template<class NodeT>
inline std::string RegionCFG<NodeT>::getFunctionName() const {
  return FunctionName;
}

template<class NodeT>
inline std::string RegionCFG<NodeT>::getRegionName() const {
  return RegionName;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
RegionCFG<NodeT>::addNode(NodeT Node, llvm::StringRef Name) {
  using BBNodeT = BasicBlockNodeT;
  BlockNodes.emplace_back(std::make_unique<BBNodeT>(this, Node, Name));
  BasicBlockNodeT *Result = BlockNodes.back().get();
  revng_log(CombLogger,
            "Building " << Name << " at address: " << Result << "\n");
  return Result;
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
RegionCFG<NodeT>::cloneNode(BasicBlockNodeT &OriginalNode) {
  using BBNodeT = BasicBlockNodeT;
  BlockNodes.emplace_back(std::make_unique<BBNodeT>(OriginalNode, this));
  BasicBlockNodeT *New = BlockNodes.back().get();
  New->setName(OriginalNode.getName().str() + " cloned");
  return New;
}

template<class NodeT>
inline void RegionCFG<NodeT>::removeNode(BasicBlockNodeT *Node) {

  revng_log(CombLogger, "Removing node named: " << Node->getNameStr() << "\n");

  for (BasicBlockNodeT *Predecessor : Node->predecessors()) {
    Predecessor->removeSuccessor(Node);
  }

  for (BasicBlockNodeT *Successor : Node->successors()) {
    Successor->removePredecessor(Node);
  }

  for (auto It = BlockNodes.begin(); It != BlockNodes.end(); It++) {
    if ((*It).get() == Node) {
      BlockNodes.erase(It);
      break;
    }
  }
}

template<class NodeT>
using BBNodeT = typename RegionCFG<NodeT>::BasicBlockNodeT;

template<class NodeT>
inline void copyNeighbors(BBNodeT<NodeT> *Dst, BBNodeT<NodeT> *Src) {
  for (BBNodeT<NodeT> *Succ : Src->successors())
    Dst->addSuccessor(Succ);
  for (BBNodeT<NodeT> *Pred : Src->predecessors())
    Dst->addPredecessor(Pred);
}

template<class NodeT>
inline void RegionCFG<NodeT>::insertBulkNodes(BasicBlockNodeTSet &Nodes,
                                              BasicBlockNodeT *Head,
                                              BBNodeMap &SubMap) {
  revng_assert(BlockNodes.empty());

  for (BasicBlockNodeT *Node : Nodes) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(*Node, this));
    BasicBlockNodeT *New = BlockNodes.back().get();
    SubMap[Node] = New;
    // The copy constructor used above does not bring along the successors and
    // the predecessors, neither adjusts the parent.
    // The following lines are a hack to fix this problem, but they momentarily
    // build a broken data structure where the predecessors and the successors
    // of the New BasicBlockNodes in *this still refer to the BasicBlockNodes in
    // the Parent CFGRegion of Nodes. This will be fixed later by updatePointers
    copyNeighbors<NodeT>(New, Node);
  }

  revng_assert(Head != nullptr);
  EntryNode = SubMap[Head];
  revng_assert(EntryNode != nullptr);
  // Fix the hack above
  for (BBNodeTUniquePtr &Node : BlockNodes)
    Node->updatePointers(SubMap);
}

template<class NodeT>
using lk_iterator = typename RegionCFG<NodeT>::links_container::iterator;

template<class NodeT>
inline llvm::iterator_range<lk_iterator<NodeT>>
RegionCFG<NodeT>::copyNodesAndEdgesFrom(RegionCFGT *O, BBNodeMap &SubMap) {
  typename links_container::difference_type NumNewNodes = 0;

  for (BasicBlockNode<NodeT> *Node : *O) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNodeT>(*Node, this));
    ++NumNewNodes;
    BasicBlockNodeT *New = BlockNodes.back().get();
    SubMap[Node] = New;
    copyNeighbors<NodeT>(New, Node);
  }

  internal_iterator BeginInserted = BlockNodes.end() - NumNewNodes;
  internal_iterator EndInserted = BlockNodes.end();
  using MovedIteratorRange = llvm::iterator_range<internal_iterator>;
  MovedIteratorRange Result = llvm::make_range(BeginInserted, EndInserted);
  for (std::unique_ptr<BasicBlockNode<NodeT>> &NewNode : Result)
    NewNode->updatePointers(SubMap);
  return Result;
}

template<class NodeT>
inline void RegionCFG<NodeT>::connectBreakNode(std::set<EdgeDescriptor> &Out,
                                               const BBNodeMap &SubMap) {
  for (EdgeDescriptor Edge : Out) {

    // Create a new break for each outgoing edge.
    BasicBlockNode<NodeT> *Break = addBreak();
    addEdge(EdgeDescriptor(SubMap.at(Edge.first), Break));
  }
}

template<class NodeT>
inline void RegionCFG<NodeT>::connectContinueNode() {
  BasicBlockNodeTVect ContinueNodes;

  // We need to pre-save the edges to avoid breaking the predecessor iterator
  for (BasicBlockNode<NodeT> *Source : EntryNode->predecessors()) {
    ContinueNodes.push_back(Source);
  }
  for (BasicBlockNode<NodeT> *Source : ContinueNodes) {

    // Create a new continue node for each retreating edge.
    BasicBlockNode<NodeT> *Continue = addContinue();
    moveEdgeTarget(EdgeDescriptor(Source, EntryNode), Continue);
  }
}

template<class NodeT>
inline std::vector<BasicBlockNode<NodeT> *>
RegionCFG<NodeT>::orderNodes(BasicBlockNodeTVect &L, bool DoReverse) {
  BasicBlockNodeTSet ToOrder;
  ToOrder.insert(L.begin(), L.end());
  llvm::ReversePostOrderTraversal<BasicBlockNode<NodeT> *> RPOT(EntryNode);
  BasicBlockNodeTVect Result;

  if (DoReverse) {
    std::reverse(RPOT.begin(), RPOT.end());
  }

  for (BasicBlockNode<NodeT> *RPOTBB : RPOT) {
    if (ToOrder.count(RPOTBB) != 0) {
      Result.push_back(RPOTBB);
    }
  }

  revng_assert(L.size() == Result.size());

  return Result;
}

template<class NodeT>
template<typename StreamT>
inline void
RegionCFG<NodeT>::streamNode(StreamT &S, const BasicBlockNodeT *BB) const {
  unsigned NodeID = BB->getID();
  S << "\"" << NodeID << "\"";
  S << " ["
    << "label=\"ID: " << NodeID << " Name: " << BB->getNameStr() << "\"";
  if (BB == EntryNode)
    S << ",fillcolor=green,style=filled";
  S << "];\n";
}

/// \brief Dump a GraphViz file on stdout representing this function
template<class NodeT>
template<typename StreamT>
inline void RegionCFG<NodeT>::dumpDot(StreamT &S) const {
  S << "digraph CFGFunction {\n";

  for (const std::unique_ptr<BasicBlockNode<NodeT>> &BB : BlockNodes) {
    streamNode(S, BB.get());
    for (auto &Successor : BB->successors()) {
      unsigned PredID = BB->getID();
      unsigned SuccID = Successor->getID();
      S << "\"" << PredID << "\""
        << " -> \"" << SuccID << "\"";
      S << " [color=green];\n";
    }
  }
  S << "}\n";
}

template<class NodeT>
inline void RegionCFG<NodeT>::dumpDotOnFile(const std::string &FileName) const {
  std::error_code EC;
  llvm::raw_fd_ostream DotFile(FileName, EC);
  revng_check(not EC, "Could not open file for printing RegionCFG dot");
  dumpDot(DotFile);
}

template<class NodeT>
inline void RegionCFG<NodeT>::dumpDotOnFile(const std::string &FolderName,
                                            const std::string &FuncName,
                                            const std::string &FileName) const {
  std::error_code EC = llvm::sys::fs::create_directory(FolderName);
  revng_check(not EC, "Could not create directory to print RegionCFG dot");
  const std::string PathName = FolderName + "/" + FuncName;
  EC = llvm::sys::fs::create_directory(PathName);
  revng_check(not EC, "Could not create directory to print RegionCFG dot");
  dumpDotOnFile(PathName + "/" + FileName);
}

template<class NodeT>
inline bool RegionCFG<NodeT>::purgeIfTrivialDummy(BBNodeT *Dummy) {
  RegionCFG<NodeT> &Graph = *this;

  revng_assert(not Dummy->isEmpty() or Dummy->predecessor_size() != 0);

  if ((Dummy->isEmpty()) and (Dummy->predecessor_size() == 1)
      and (Dummy->successor_size() == 1)) {

    revng_log(CombLogger, "Purging dummy node " << Dummy->getNameStr());

    BasicBlockNode<NodeT> *Predecessor = Dummy->getPredecessorI(0);
    BasicBlockNode<NodeT> *Successor = Dummy->getSuccessorI(0);

    // Connect directly predecessor and successor, and remove the dummy node
    // under analysis
    moveEdgeTarget({ Predecessor, Dummy }, Successor);
    Graph.removeNode(Dummy);
    return true;
  }

  return false;
}

template<class NodeT>
inline bool RegionCFG<NodeT>::purgeTrivialDummies() {
  RegionCFG<NodeT> &Graph = *this;
  bool RemovedNow = true;
  bool Removed = false;

  while (RemovedNow) {
    RemovedNow = false;
    for (auto *Node : Graph) {
      RemovedNow = purgeIfTrivialDummy(Node);
      if (RemovedNow) {
        Removed = true;
        break;
      }
    }
  }

  return Removed;
}

template<class NodeT>
inline void RegionCFG<NodeT>::purgeVirtualSink(BasicBlockNode<NodeT> *Sink) {

  RegionCFG<NodeT> &Graph = *this;

  BasicBlockNodeTVect WorkList;
  BasicBlockNodeTVect PurgeList;

  WorkList.push_back(Sink);

  while (!WorkList.empty()) {
    BasicBlockNode<NodeT> *CurrentNode = WorkList.back();
    WorkList.pop_back();

    if (CurrentNode->isEmpty()) {
      PurgeList.push_back(CurrentNode);

      for (BasicBlockNode<NodeT> *Predecessor : CurrentNode->predecessors()) {
        WorkList.push_back(Predecessor);
      }
    }
  }

  for (BasicBlockNode<NodeT> *Purge : PurgeList) {
    Graph.removeNode(Purge);
  }
}

template<class NodeT>
inline std::vector<BasicBlockNode<NodeT> *>
RegionCFG<NodeT>::getInterestingNodes(BasicBlockNodeT *Cond) {

  RegionCFG<NodeT> &Graph = *this;

  // Retrieve the immediate postdominator.
  llvm::DomTreeNodeBase<BasicBlockNode<NodeT>> *PostBase = PDT[Cond]->getIDom();
  BasicBlockNode<NodeT> *PostDominator = PostBase->getBlock();

  BasicBlockNodeTSet Candidates = findReachableNodes(*Cond, *PostDominator);

  BasicBlockNodeTVect NotDominatedCandidates;
  for (BasicBlockNode<NodeT> *Node : Candidates) {
    if (!DT.dominates(Cond, Node)) {
      NotDominatedCandidates.push_back(Node);
    }
  }

  // TODO: Check that this is the order that we want.
  NotDominatedCandidates = Graph.orderNodes(NotDominatedCandidates, true);

  return NotDominatedCandidates;
}

inline bool isGreater(unsigned Op1, unsigned Op2) {
  unsigned MultiplicativeFactor = 1;
  if (Op1 > (MultiplicativeFactor * Op2)) {
    return true;
  } else {
    return false;
  }
}

template<class NodeT>
inline BasicBlockNode<NodeT> *
RegionCFG<NodeT>::cloneUntilExit(BasicBlockNode<NodeT> *Node,
                                 BasicBlockNode<NodeT> *Sink) {

  // Clone the postdominator node.
  BBNodeMap CloneMap;
  BasicBlockNode<NodeT> *Clone = cloneNode(*Node);

  // Insert the postdominator clone in the map.
  CloneMap[Node] = Clone;

  BasicBlockNodeTVect WorkList;
  WorkList.push_back(Node);

  // Set of nodes which have been already processed.
  BasicBlockNodeTSet AlreadyProcessed;

  while (!WorkList.empty()) {
    BasicBlockNode<NodeT> *CurrentNode = WorkList.back();
    WorkList.pop_back();

    // Ensure that we are not processing the sink node.
    revng_assert(CurrentNode != Sink);

    if (AlreadyProcessed.count(CurrentNode) == 0) {
      AlreadyProcessed.insert(CurrentNode);
    } else {
      continue;
    }

    // Get the clone of the `CurrentNode`.
    BasicBlockNode<NodeT> *CurrentClone = CloneMap.at(CurrentNode);

    for (BasicBlockNode<NodeT> *Successor : CurrentNode->successors()) {
      // If the successor is not the sink, create and edge that directly
      // connects it.
      if (Successor != Sink) {
        BasicBlockNode<NodeT> *SuccessorClone = nullptr;

        // The clone of the successor node already exists.
        if (CloneMap.count(Successor)) {
          SuccessorClone = CloneMap.at(Successor);
        } else {

          // The clone of the successor does not exist, create it in place.
          SuccessorClone = cloneNode(*Successor);
          CloneMap[Successor] = SuccessorClone;
        }

        // Create the edge to the clone of the successor.
        revng_assert(SuccessorClone != nullptr);
        addEdge(EdgeDescriptor(CurrentClone, SuccessorClone));

        // Add the successor to the worklist.
        WorkList.push_back(Successor);
      }
    }
  }

  return Clone;
}

template<class NodeT>
inline void RegionCFG<NodeT>::untangle() {
  // TODO: Here we handle only conditional nodes with two successors. We should
  //       consider extending the untangle procedure also to conditional nodes
  //       with more than two successors (switch nodes).

  revng_assert(isDAG());

  RegionCFG<NodeT> &Graph = *this;

  // Collect all the conditional nodes in the graph.
  BasicBlockNodeTVect ConditionalNodes;
  for (auto It = Graph.begin(); It != Graph.end(); It++) {
    if ((*It)->successor_size() == 2) {
      ConditionalNodes.push_back(*It);
    }
  }

  // Collect entry and exit nodes.
  BasicBlockNodeTVect ExitNodes;
  for (auto It = Graph.begin(); It != Graph.end(); It++) {
    if ((*It)->successor_size() == 0) {
      ExitNodes.push_back(*It);
    }
  }

  // Add a new virtual sink node to computer the postdominator.
  BasicBlockNode<NodeT> *Sink = Graph.addArtificialNode("Sink");
  for (BasicBlockNode<NodeT> *Exit : ExitNodes) {
    addEdge(EdgeDescriptor(Exit, Sink));
  }

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("untangle",
                        FunctionName,
                        "Region-" + RegionName + "-initial-state");
  }

  // Map which contains the precomputed wheight for each node in the graph. In
  // case of a code node the weight will be equal to the number of instruction
  // in the original basic block; in case of a collapsed node the weight will be
  // the sum of the weights of all the nodes contained in the collapsed graph.
  std::map<BasicBlockNode<NodeT> *, size_t> WeightMap;
  for (BasicBlockNode<NodeT> *Node : Graph.nodes()) {
    WeightMap[Node] = Node->getWeight();
  }

  std::set<EdgeDescriptor> InlinedEdges;

  // Order the conditional nodes in postorder.
  ConditionalNodes = Graph.orderNodes(ConditionalNodes, false);

  while (!ConditionalNodes.empty()) {
    if (CombLogger.isEnabled()) {
      Graph.dumpDotOnFile("untangle",
                          FunctionName,
                          "Region-" + RegionName + "-debug");
    }
    BasicBlockNode<NodeT> *Conditional = ConditionalNodes.back();
    ConditionalNodes.pop_back();

    // Update the information of the dominator and postdominator trees.
    DT.recalculate(Graph);

    for (EdgeDescriptor Edge : InlinedEdges)
      removeEdge(Edge);

    PDT.recalculate(Graph);

    // Reattach the edges disconnected for the PDT computation.
    for (EdgeDescriptor Edge : InlinedEdges)
      addEdge(Edge);

    // Update the postdominator
    BasicBlockNodeT *PostDominator = PDT[Conditional]->getIDom()->getBlock();
    revng_assert(PostDominator != nullptr);

    // Ensure that we have both the successors.
    revng_assert(Conditional->successor_size() == 2);

    // Get the first node of the then and else branches respectively.
    // TODO: Check that this is the right way to do this. At this point we
    //       cannot assume that we have the `getThen()` and `getFalse()`
    //       methods.
    BasicBlockNode<NodeT> *ThenChild = Conditional->getSuccessorI(0);
    BasicBlockNode<NodeT> *ElseChild = Conditional->getSuccessorI(1);

    // Collect all the nodes laying between the branches
    BasicBlockNodeTSet ThenNodes = findReachableNodes(*ThenChild,
                                                      *PostDominator);

    BasicBlockNodeTSet ElseNodes = findReachableNodes(*ElseChild,
                                                      *PostDominator);

    // Remove the postdominator from both the sets.
    ThenNodes.erase(PostDominator);
    ElseNodes.erase(PostDominator);

    const auto EdgeDominates = [DT = &DT](const EdgeDescriptor &E,
                                          BasicBlockNodeT *N) {
      const auto &[Src, Dst] = E;

      if (not DT->dominates(Dst, N))
        return false;

      if (Dst->predecessor_size() < 2)
        return true;

      bool DuplicateEdge = false;
      for (BasicBlockNodeT *Pred : Dst->predecessors()) {
        if (Pred == Src) {
          if (DuplicateEdge)
            return false;
          DuplicateEdge = true;
          continue;
        }

        if (not DT->dominates(Dst, Pred))
          return false;
      }
      return true;
    };

    // New implementation of the dominance criterion which uses the then and
    // else edges to compute the dominance.
    if (EdgeDominates({ Conditional, ElseChild }, ElseChild)) {
      const auto DominatedByElse = [DT = &DT, ElseChild](auto *Node) {
        return DT->dominates(ElseChild, Node);
      };
      // TODO: substitute the following loop with std::set::erase_if when it
      // becomes available.
      for (auto I = ElseNodes.begin(), E = ElseNodes.end(); I != E;) {
        if (DominatedByElse(*I))
          I = ElseNodes.erase(I);
        else
          ++I;
      }
    }

    if (EdgeDominates({ Conditional, ThenChild }, ThenChild)) {
      const auto DominatedByThen = [DT = &DT, ThenChild](auto *Node) {
        return DT->dominates(ThenChild, Node);
      };
      // TODO: substitute the following loop with std::set::erase_if when it
      // becomes available.
      for (auto I = ThenNodes.begin(), E = ThenNodes.end(); I != E;) {
        if (DominatedByThen(*I))
          I = ThenNodes.erase(I);
        else
          ++I;
      }
    }

    // Compute the weight of the `then` and `else` branches.
    unsigned ThenWeight = 0;
    unsigned ElseWeight = 0;

    for (BasicBlockNode<NodeT> *Node : ThenNodes) {
      ThenWeight += WeightMap[Node];
    }

    for (BasicBlockNode<NodeT> *Node : ElseNodes) {
      ElseWeight += WeightMap[Node];
    }

    // The weight of the nodes placed after the immediate postdominator is the
    // sum of all the weights of the nodes which are reachable starting from the
    // immediate post dominator and the sink node (to which all the exits have
    // been connected).
    unsigned PostDominatorWeight = 0;
    BasicBlockNodeTSet PostDominatorToExit = findReachableNodes(*PostDominator,
                                                                *Sink);

    for (BasicBlockNode<NodeT> *Node : PostDominatorToExit) {
      PostDominatorWeight += WeightMap[Node];
    }

    // Criterion which decides if we can apply the untangle optimization to the
    // conditional under analysis.
    // We define 3 weights:
    // - 1) weight(then) + weight(else)
    // - 2) weight(then) + weight(postdom)
    // - 3) weight(else) + weight(postdom)
    //
    // We need to operate the split if:
    // 2 >> 3
    // 1 >> 3
    // and specifically we need to split the `else` branch.
    //
    // We need to operate the split if:
    // 3 >> 2
    // 1 >> 2
    // and specifically we need to split the `then` branch.
    //
    // We can also define in a dynamic way the >> operator, so we can change the
    // threshold that triggers the split.

    unsigned CombingCost = ThenWeight + ElseWeight;
    unsigned UntangleThenCost = ThenWeight + PostDominatorWeight;
    unsigned UntangleElseCost = ElseWeight + PostDominatorWeight;
    unsigned UntanglingCost = std::min(UntangleThenCost, UntangleElseCost);

    if (isGreater(CombingCost, UntanglingCost)) {
      revng_log(CombLogger, FunctionName << ":");
      revng_log(CombLogger, RegionName << ":");
      revng_log(CombLogger,
                "Found untangle candidate " << Conditional->getNameStr());
      revng_log(CombLogger, "CombingCost:" << CombingCost);
      revng_log(CombLogger, "UntangleThenCost:" << UntangleThenCost);
      revng_log(CombLogger, "UntangleElseCost:" << UntangleElseCost);

      // Register a tentative untangle in the dedicated counter.
      UntangleTentativeCounter++;

      // Register an actual untangle in the dedicated counter.
      UntanglePerformedCounter++;
      revng_log(CombLogger, "Actually splitting node");

      auto *ToUntangle = (UntangleThenCost > UntangleElseCost) ? ElseChild :
                                                                 ThenChild;
      // Perform the split from the first node of the then/else branches.
      // We fully inline all the nodes belonging to the branch we are untangling
      // till the exit node.
      BasicBlockNode<NodeT> *UntangledChild = cloneUntilExit(ToUntangle, Sink);

      // Move the edge coming out of the conditional node to the new clone of
      // the node.
      moveEdgeTarget(EdgeDescriptor(Conditional, ToUntangle), UntangledChild);

      // Updated the information about the inlining edge.
      InlinedEdges.erase(EdgeDescriptor(Conditional, ToUntangle));
      InlinedEdges.insert(EdgeDescriptor(Conditional, UntangledChild));

      // Remove nodes that have no predecessors (nodes that are the result of
      // node cloning and that remains dandling around).
      // While doing this, update InlineEdges.
      bool Removed = true;
      while (Removed) {
        Removed = false;
        BasicBlockNode<NodeT> *Entry = &getEntryNode();
        for (auto It = begin(); It != end(); ++It) {
          if ((Entry != *It and (*It)->predecessor_size() == 0)) {
            // TODO: substitute the following loop with std::set::erase_if when
            // it becomes available.
            for (auto I = InlinedEdges.begin(), E = InlinedEdges.end();
                 I != E;) {
              if (I->first == *It or I->second == *It)
                I = InlinedEdges.erase(I);
              else
                ++I;
            }
            removeNode(*It);
            Removed = true;
            break;
          }
        }
      }
    }
  }

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("untangle",
                        FunctionName,
                        "Region-" + RegionName + "-after-processing");
  }

  // Remove the sink node.
  purgeVirtualSink(Sink);

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("untangle",
                        FunctionName,
                        "Region-" + RegionName + "-after-sink-removal");
  }
}

template<class NodeT>
inline void RegionCFG<NodeT>::inflate() {

  // Call the untangle preprocessing.
  untangle();

  revng_assert(isDAG());

  // Apply the comb to a RegionCFG object.
  // TODO: handle all the collapsed regions.
  RegionCFG<NodeT> &Graph = *this;

  // Collect entry and exit nodes.
  BasicBlockNodeTVect ExitNodes;
  for (auto *Node : Graph) {
    if (Node->successor_size() == 0) {
      ExitNodes.push_back(Node);
    }
  }

  BasicBlockNode<NodeT> *Entry = &Graph.getEntryNode();
  if (CombLogger.isEnabled()) {
    CombLogger << "The entry node is:\n";
    CombLogger << Entry->getNameStr() << "\n";
    CombLogger << "In the graph the exit nodes are:\n";
    for (BasicBlockNode<NodeT> *Node : ExitNodes) {
      CombLogger << Node->getNameStr() << "\n";
    }
  }

  // Helper data structure for exit reachability computation.
  std::map<BasicBlockNode<NodeT> *, BasicBlockNodeTSet> ReachableExits;

  // Collect nodes reachable from each exit node in the graph.
  for (BasicBlockNode<NodeT> *Exit : ExitNodes) {
    revng_log(CombLogger, "From exit node: " << Exit->getNameStr());
    revng_log(CombLogger, "We can reach:");
    for (BasicBlockNode<NodeT> *Node : llvm::inverse_depth_first(Exit)) {
      revng_log(CombLogger, Node->getNameStr());
      ReachableExits[Node].insert(Exit);
    }
  }

  // Dump graph before virtual sink add.
  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-before-sink");
  }

  // Add a new virtual sink node to which all the exit nodes are connected.
  BasicBlockNode<NodeT> *Sink = Graph.addArtificialNode("Sink");
  for (BasicBlockNode<NodeT> *Exit : ExitNodes) {
    addEdge(EdgeDescriptor(Exit, Sink));
  }

  // Dump graph after virtual sink add.
  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-after-sink");
  }

  // Refresh information of dominator tree.
  DT.recalculate(Graph);

  // Collect all the conditional nodes in the graph.
  // This is the working list of conditional nodes on which we will operate and
  // will contain only the filtered conditionals.
  BasicBlockNodeTVect ConditionalNodes;

  for (BBNodeT *Node : Graph) {
    switch (Node->successor_size()) {
    case 0:
    case 1:
      // We don't need to add it to the conditional nodes vector.
      break;
    case 2: {

      BasicBlockNodeTSet ThenExits = ReachableExits[Node->getSuccessorI(0)];
      BasicBlockNodeTSet ElseExits = ReachableExits[Node->getSuccessorI(1)];

      // If the exit nodes reachable from the Then and from the Else are not
      // disjoint, then we add Node to ConditionalNodes because it can induce
      // duplication.
      if (not disjoint(ThenExits, ElseExits)) {
        ConditionalNodes.push_back(Node);
        break;
      }

      // Check that we do not dominate at maximum on of the two sets of
      // reachable exits.
      bool ThenIsDominated = true;
      bool ElseIsDominated = true;
      for (BasicBlockNode<NodeT> *Exit : ThenExits) {
        if (not DT.dominates(Node, Exit)) {
          ThenIsDominated = false;
          break;
        }
      }
      for (BasicBlockNode<NodeT> *Exit : ElseExits) {
        if (not DT.dominates(Node, Exit)) {
          ElseIsDominated = false;
          break;
        }
      }

      // If there is one set of exits that Node entirely dominates, we can
      // blacklist it because it will never cause duplication.
      // The reason is that the set of exits that we dominate can be completely
      // inlined and absorbed either into the then or into the else.
      if (ThenIsDominated or ElseIsDominated) {
        revng_log(CombLogger,
                  "Blacklisted conditional: " << Node->getNameStr());
      } else {
        ConditionalNodes.push_back(Node);
      }
    } break;
    default: {
      // We are in presence of a switch node, which should be considered as a
      // conditional node for what concerns the combing stage.
      ConditionalNodes.push_back(Node);
    }
    }
  }

  ConditionalNodes = Graph.orderNodes(ConditionalNodes, false);

  if (CombLogger.isEnabled()) {
    CombLogger << "Conditional nodes present in the graph are:\n";
    for (BasicBlockNode<NodeT> *Node : ConditionalNodes) {
      CombLogger << Node->getNameStr() << "\n";
    }
  }

  // Equivalence-class like set to keep track of all the cloned nodes created
  // starting from an original node.
  std::map<BasicBlockNode<NodeT> *, SmallPtrSet<NodeT>> NodesEquivalenceClass;

  // Map to keep track of the cloning relationship.
  BBNodeMap CloneToOriginalMap;

  // Initialize a list containing the reverse post order of the nodes of the
  // graph.
  std::list<BasicBlockNode<NodeT> *> RevPostOrderList;
  llvm::ReversePostOrderTraversal<BasicBlockNode<NodeT> *> RPOT(Entry);
  for (BasicBlockNode<NodeT> *RPOTBB : RPOT) {
    RevPostOrderList.push_back(RPOTBB);
    NodesEquivalenceClass[RPOTBB].insert(RPOTBB);
    CloneToOriginalMap[RPOTBB] = RPOTBB;
  }

  // Refresh information of dominator and postdominator trees.
  DT.recalculate(Graph);
  PDT.recalculate(Graph);

  // Map to retrieve the post dominator for each conditional node.
  BBNodeMap CondToPostDomMap;

  // Compute the immediate post-dominator for each conditional node.
  for (BasicBlockNode<NodeT> *Conditional : ConditionalNodes) {
    BasicBlockNode<NodeT> *PostDom = PDT[Conditional]->getIDom()->getBlock();
    revng_assert(PostDom != nullptr);
    CondToPostDomMap[Conditional] = PostDom;
  }

  while (not ConditionalNodes.empty()) {

    // Process each conditional node after ordering it.
    BasicBlockNode<NodeT> *Conditional = ConditionalNodes.back();
    revng_assert(Conditional != Sink);
    ConditionalNodes.pop_back();

    // Retrieve a reference to the set of postdominators.
    auto PostDomIt = CondToPostDomMap.find(Conditional);
    revng_assert(PostDomIt != CondToPostDomMap.end());
    auto PostDomSetIt = NodesEquivalenceClass.find(PostDomIt->second);
    revng_assert(PostDomSetIt != NodesEquivalenceClass.end());

    if (CombLogger.isEnabled()) {
      revng_log(CombLogger,
                "Analyzing conditional node " << Conditional->getNameStr());
      Graph.dumpDotOnFile("inflates",
                          FunctionName,
                          "Region-" + RegionName + "-conditional-"
                            + Conditional->getNameStr() + "-begin");
    }

    // List to keep track of the nodes that we still need to analyze.
    SmallPtrSet<NodeT> WorkList;
    // Enqueue in the worklist the successors of the contional node.
    for (BasicBlockNode<NodeT> *Successor : Conditional->successors())
      WorkList.insert(Successor);

    // Keep a set of the visited nodes for the current conditional node.
    SmallPtrSet<NodeT> Visited = { Conditional };

    // Get an iterator from the reverse post order list in the position of the
    // conditional node.
    auto ListIt = std::find(RevPostOrderList.begin(),
                            RevPostOrderList.end(),
                            Conditional);
    revng_assert(ListIt != RevPostOrderList.end());

    int Iteration = 0;
    while (++ListIt != RevPostOrderList.end() and not WorkList.empty()) {

      if (*ListIt == Sink) {
        revng_assert(std::next(ListIt) == RevPostOrderList.end());
        continue;
      }

      // Scan the working list and the reverse post order in a parallel manner.
      if (not WorkList.count(*ListIt))
        continue; // Go to the next node in reverse postorder.

      // Otherwise this node is in the worklist, and we have to analyze it.
      BasicBlockNode<NodeT> *Candidate = *ListIt;
      revng_assert(Candidate != nullptr);

      revng_log(CombLogger, "Analyzing candidate " << Candidate->getNameStr());

      bool AllPredAreVisited = std::all_of(Candidate->predecessors().begin(),
                                           Candidate->predecessors().end(),
                                           [&Visited](auto *Pred) {
                                             return Visited.count(Pred);
                                           });
      WorkList.erase(Candidate);
      Visited.insert(Candidate);

      // Postdom flag, which is useful to understand if the dummies we will
      // insert will need to substitute the current postdominator.
      bool IsPostDom = PostDomSetIt->second.count(Candidate);

      if (not IsPostDom) {
        for (BasicBlockNode<NodeT> *Successor : Candidate->successors())
          WorkList.insert(Successor);
      } else {
        revng_log(CombLogger,
                  Candidate->getNameStr()
                    << " is Post-Dominator of " << Conditional->getNameStr());
      }

      if (AllPredAreVisited)
        continue; // Go to the next node in reverse postorder.

      // Decide wether to insert a dummy or to duplicate.
      if (IsPostDom and Candidate->predecessor_size() > 2) {

        BasicBlockNodeTVect NewDummyPredecessors;
        revng_log(CombLogger, "Current predecessors are:");
        for (BasicBlockNode<NodeT> *Predecessor : Candidate->predecessors()) {
          revng_log(CombLogger, Predecessor->getNameStr());
          if (Visited.count(Predecessor))
            NewDummyPredecessors.push_back(Predecessor);
        }
        // We don't insert the dummy, because it would be a dummy with a single
        // predecessor and a single successor, which is pointless.
        if (NewDummyPredecessors.size() < 2)
          continue;

        revng_log(CombLogger,
                  "Inserting a dummy node for " << Candidate->getNameStr());

        // Insert a dummy node. Notice, this is guaranteed not to be trivial
        // because it will have more than one predecessor.
        BasicBlockNode<NodeT> *Dummy = Graph.addArtificialNode();

        for (BasicBlockNode<NodeT> *Predecessor : NewDummyPredecessors) {
          revng_log(CombLogger,
                    "Moving edge from predecessor " << Predecessor->getNameStr()
                                                    << " to dummy");
          moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate), Dummy);
        }

        addEdge(EdgeDescriptor(Dummy, Candidate));

        // Remove from the visited set the node which triggered the creation of
        // the dummy nodes, because we're not really analyzing it now, since
        // we're just inserting the dummy.
        // For the same reason we re-insert it in the WorkList, otherwise it
        // will be skipped at the next iteration.
        Visited.erase(Candidate);
        WorkList.insert(Candidate);

        // The new dummy node does not lead back to any original node, for this
        // reason we need to insert a new entry in the `CloneToOriginalMap`.
        CloneToOriginalMap[Dummy] = Dummy;

        revng_log(CombLogger,
                  "Update conditional post-dominator. Old: "
                    << PostDomIt->second->getNameStr()
                    << " New: " << Dummy->getNameStr());

        // The dummy is now the post dominator of conditional
        PostDomIt->second = Dummy;
        PostDomSetIt = NodesEquivalenceClass.insert({ Dummy, { Dummy } }).first;

        // Mark the dummy to explore.
        WorkList.insert(Dummy);

        // Insert the dummy nodes in the reverse post order list. The insertion
        // order is particularly relevant, because we have added a dummy that
        // now post-dominates the region starting from Conditional, while
        // Candidate (which is the post-dominator of Conditional here), is a
        // successor of Dummy. Hence Dummy must come first in reverse post
        // order, otherwise future RPOT visits based on RevPostOrderList might
        // be disrupted.
        auto PrevListIt = std::prev(ListIt);
        RevPostOrderList.insert(ListIt, Dummy);
        ListIt = PrevListIt;

      } else {

        // Duplicate node.
        DuplicationCounter++;
        revng_log(CombLogger, "Duplicating node " << Candidate->getNameStr());

        BasicBlockNode<NodeT> *Duplicated = Graph.cloneNode(*Candidate);
        revng_assert(Duplicated != nullptr);

        // Initialize the successors of the Duplicated node with the same
        // successors of Candidate node
        for (BasicBlockNode<NodeT> *Successor : Candidate->successors())
          addEdge(EdgeDescriptor(Duplicated, Successor));

        bool Same = Candidate->successor_size() == Duplicated->successor_size();
        revng_assert(Same);

        // Move Candidate's predecessors that have not been visited yet, so that
        // they become predecessors of Duplicated
        BasicBlockNodeTVect NotVisitedPredecessors;
        for (BasicBlockNode<NodeT> *Predecessor : Candidate->predecessors())
          if (not Visited.count(Predecessor))
            NotVisitedPredecessors.push_back(Predecessor);

        for (BasicBlockNode<NodeT> *Predecessor : NotVisitedPredecessors) {
          moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate), Duplicated);
          revng_log(CombLogger,
                    "Moving edge from predecessor "
                      << Predecessor->getNameStr() << " to "
                      << Duplicated->getNameStr());
        }

        if (CombLogger.isEnabled()) {
          Graph.dumpDotOnFile("inflates",
                              FunctionName,
                              "Region-" + RegionName + "-before-purge-dummies-"
                                + Conditional->getNameStr() + "-"
                                + std::to_string(Iteration));
        }

        BasicBlockNode<NodeT> *OriginalNode = CloneToOriginalMap.at(Candidate);

        bool AreDummies = Candidate->isEmpty();
        revng_assert(AreDummies == Duplicated->isEmpty());
        if (AreDummies) {
          revng_log(CombLogger, "Duplicated is dummy");
          revng_assert(Candidate->successor_size() < 2
                       and Duplicated->successor_size() < 2);

          // Notice: after this call Duplicated is invalid if the call returns
          // true, meaning that dereferncing it is bad. You can still use it as
          // a key or value into maps though.
          if (not purgeIfTrivialDummy(Duplicated)) {
            // Add the cloned node in the equivalence class of the original
            // node.
            CloneToOriginalMap[Duplicated] = OriginalNode;
            NodesEquivalenceClass.at(OriginalNode).insert(Duplicated);

            // If it wasn't purged, insert the cloned node in the reverse post
            // order list. Here the order is not strictly relevant, because
            // there is no strict relationship between Candidate and Duplicated.
            RevPostOrderList.insert(ListIt, Duplicated);
          } else {
            revng_log(CombLogger, "Duplicated is trivial");
          }

          // The duplication process divides the edges incoming to Candidate,
          // and it moves some of them to Duplicated. If Candidate is a dummy
          // node, this process may make it trivial. In that case we want to
          // remove it.

          // Notice: after this call Candidate is invalid if the call returns
          // true, meaning that dereferncing it is bad. You can still use it as
          // a key or value into maps though.
          if (purgeIfTrivialDummy(Candidate)) {
            revng_log(CombLogger, "Candidate is now trivial");
            CloneToOriginalMap.erase(Candidate);
            NodesEquivalenceClass.at(OriginalNode).erase(Candidate);
            Visited.erase(Candidate);
            // Erase Candidate from the post order list, but update ListIt so
            // that after the removal it points to the element that was
            // previously before Candidate. In this way, at the next iteration
            // of the loop on RevPostOrderList we go on with the element that
            // was right after Candidate before its removal.
            auto PrevListIt = std::prev(ListIt);
            RevPostOrderList.erase(ListIt);
            ListIt = PrevListIt;
          }

        } else {
          revng_log(CombLogger, "Duplicated is not dummy");

          // Add the cloned node in the equivalence class of the original node.
          CloneToOriginalMap[Duplicated] = OriginalNode;
          NodesEquivalenceClass.at(OriginalNode).insert(Duplicated);

          // If the node we are duplicating is a conditional node, add it to the
          // vector of ConditionalNodes. Remember, that vector is ordered in
          // post-order, and here we're iterating in reverse-post-order (thanks
          // to RevPostOrderList) so pushing back into ConditionalNodes always
          // preserves the property that ConditionalNodes is sorted in
          // post-order.
          if (auto It = CondToPostDomMap.find(Candidate);
              It != CondToPostDomMap.end()) {
            CondToPostDomMap[Duplicated] = It->second;
            ConditionalNodes.push_back(Duplicated);
          }

          // Insert the cloned node in the reverse post order list, right before
          // the Candidate. This is not important right now, because we don't
          // add it to the WorkList. It will become important if whenever
          // Duplicated is reached with a traversal based on RevPostOrderList
          // starting from a different Conditional.
          // In this sense, it's not really important to insert Duplicated
          // before or after Candidate, since they have no strict relationship
          // in the reverse post order.
          RevPostOrderList.insert(ListIt, Duplicated);
        }
      }

      if (CombLogger.isEnabled()) {
        Graph.dumpDotOnFile("inflates",
                            FunctionName,
                            "Region-" + RegionName + "-conditional-"
                              + Conditional->getNameStr() + "-"
                              + std::to_string(Iteration));
      }
      Iteration++;
    }

    revng_log(CombLogger,
              "Finished looking at conditional: " << Conditional->getNameStr());
  }

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName
                          + "-before-final-inflate-cleanup");
  }

  // Purge extra dummy nodes introduced.
  purgeTrivialDummies();
  purgeVirtualSink(Sink);

  if (CombLogger.isEnabled()) {
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-after-combing");
  }
}

template<class NodeT>
inline bool isASwitch(BasicBlockNode<NodeT> *Node) {

  // TODO: remove this workaround for searching for switch nodes.
  if (Node->getOriginalNode()) {
    llvm::BasicBlock *OriginalBB = Node->getOriginalNode();
    llvm::Instruction *TerminatorBB = OriginalBB->getTerminator();
    return llvm::isa<llvm::SwitchInst>(TerminatorBB);
  }

  // The node may be an artifical node, therefore not an original switch.
  return false;
}

template<class NodeT>
inline void RegionCFG<NodeT>::generateAst() {

  RegionCFG<NodeT> &Graph = *this;

  // Apply combing to the current RegionCFG.
  if (ToInflate) {

    if (CombLogger.isEnabled()) {
      CombLogger << "Weaveing region " + RegionName + "\n";
      dumpDotOnFile("weaves", FunctionName, "PREWEAVE");
    }

    // Invoke the weave function.
    Graph.weave();

    if (CombLogger.isEnabled()) {
      dumpDotOnFile("weaves", FunctionName, "POSTWEAVE");

      CombLogger << "Inflating region " + RegionName + "\n";
      dumpDotOnFile("dots", FunctionName, "PRECOMB");
    }

    Graph.inflate();
    ToInflate = false;
    if (CombLogger.isEnabled()) {
      dumpDotOnFile("dots", FunctionName, "POSTCOMB");
    }
  }

  // TODO: factorize out the AST generation phase.
  llvm::DominatorTreeBase<BasicBlockNode<NodeT>, false> ASTDT;
  ASTDT.recalculate(Graph);
  ASTDT.updateDFSNumbers();

  CombLogger << DoLog;

  std::map<unsigned, BasicBlockNode<NodeT> *> DFSNodeMap;

  // Compute the ideal order of visit for performing the weaving.
  for (BasicBlockNode<NodeT> *Node : Graph.nodes()) {
    DFSNodeMap[ASTDT[Node]->getDFSNumOut()] = Node;
  }

  // Visiting order of the dominator tree.
  if (CombLogger.isEnabled()) {
    for (auto &Pair : DFSNodeMap) {
      CombLogger << Pair.second->getNameStr() << "\n";
    }
  }

  for (auto &Pair : DFSNodeMap) {
    BasicBlockNode<NodeT> *Node = Pair.second;

    // Collect the children nodes in the dominator tree.
    std::vector<llvm::DomTreeNodeBase<BasicBlockNode<NodeT>> *>
      Children = ASTDT[Node]->getChildren();

    std::vector<ASTNode *> ASTChildren;
    BasicBlockNodeTVect BBChildren;
    for (llvm::DomTreeNodeBase<BasicBlockNode<NodeT>> *TreeNode : Children) {
      BasicBlockNode<NodeT> *BlockNode = TreeNode->getBlock();
      ASTNode *ASTPointer = AST.findASTNode(BlockNode);
      ASTChildren.push_back(ASTPointer);
      BBChildren.push_back(BlockNode);
    }

    // Check that the two vector have the same size.
    revng_assert(Children.size() == ASTChildren.size());
    using UniqueASTNode = ASTTree::ast_unique_ptr;

    // Handle collapsded node.
    UniqueASTNode ASTObject;
    if (Node->isCollapsed()) {
      revng_assert(ASTChildren.size() <= 1);
      RegionCFG<NodeT> *BodyGraph = Node->getCollapsedCFG();
      revng_assert(BodyGraph != nullptr);
      revng_log(CombLogger,
                "Inspecting collapsed node: " << Node->getNameStr());
      BodyGraph->generateAst();
      if (ASTChildren.size() == 1) {
        ASTNode *Body = BodyGraph->getAST().getRoot();
        ASTObject.reset(new ScsNode(Node, Body, ASTChildren[0]));
      } else {
        ASTNode *Body = BodyGraph->getAST().getRoot();
        ASTObject.reset(new ScsNode(Node, Body));
      }
    } else if (Node->isDispatcher() or isASwitch(Node)) {
      // This should be dedicated to handle switch node. Unfortunately not all
      // the switch nodes are guaranteed to have more than 3 dominated nodes.
      revng_assert(not Node->isBreak() and not Node->isContinue()
                   and not Node->isSet());

      // Assert that in this case we are in presence of a switch instruction.
      revng_assert(Node->isCode() or Node->isDispatcher());

      // If we are in presence of a dispatcher, we do not have a corresponding
      // switch instruction in the LLVM ir.
      llvm::Value *SwitchValue = nullptr;
      if (!Node->isDispatcher()) {

        llvm::BasicBlock *OriginalNode = Node->getOriginalNode();
        llvm::Instruction *Terminator = OriginalNode->getTerminator();
        llvm::SwitchInst *Switch = llvm::cast<llvm::SwitchInst>(Terminator);

        SwitchValue = Switch->getCondition();
      }

      // Collect the successor, if present at all.
      // We should have at maximum a single node which is directly dominated
      // by the head of the switch, but which is not reachable from the
      // switch head.
      // TODO: This could be not true if for example the switch head node is
      //       directly connected to the postdominator.
      // TODO: Elect the real immediate postdominator.
      // Fill the cases container with the ASTNodes pointing to the cases
      BasicBlockNode<NodeT> *PostDomBBNode = nullptr;
      ASTNode *PostDomASTNode = nullptr;
      unsigned NodeI = 0;
      for (BasicBlockNode<NodeT> *Dominated : BBChildren) {
        if (not Node->hasSuccessor(Dominated)) {
          revng_assert(PostDomBBNode == nullptr);
          revng_assert(PostDomASTNode == nullptr);
          PostDomBBNode = Dominated;

          // Retrieve the corresponding ASTNode using the index we computed.
          PostDomASTNode = ASTChildren.at(NodeI);
        }

        // Increment the index.
        NodeI++;
      }

      // Build the case vector depending on the fact that we are building a
      // regular or dispatcher switch.
      RegularSwitchNode::case_value_container CaseValuesRegular;
      SwitchDispatcherNode::case_value_container CaseValuesCheck;
      ASTNode *DefaultASTNode = nullptr;
      if (!Node->isDispatcher()) {
        llvm::BasicBlock *OriginalNode = Node->getOriginalNode();
        llvm::Instruction *Terminator = OriginalNode->getTerminator();
        llvm::SwitchInst *Switch = llvm::cast<llvm::SwitchInst>(Terminator);
        unsigned Index = 0;
        for (ASTNode *A : ASTChildren) {
          if (A != PostDomASTNode) {
            BBNodeT *N = BBChildren[Index];
            llvm::BasicBlock *B = N->getOriginalNode();
            revng_assert(N->isCode() or N->isDispatcher());

            // In presence of a dispatcher node (which can be a symptom of a
            // weaving pass) we need to collect all the cases from the
            // underlying nodes.
            llvm::SmallPtrSet<llvm::ConstantInt *, 1> CaseSet;
            if (N->isDispatcher()) {

              // The dispatcher node has no correspondence in terms of
              // original IR. But if we find a dispatcher, these means that a
              // weaving process has took place, and therefore we need to go
              // over the children of the dispatcher node in order to find the
              // original nodes which gave origin to the dispatcher node.
              // We then iterate over them to collect the case values, and we
              // put them in the `CaseSet`, which will be expanded to an `or`
              // of all the values during the emission of C.
              revng_assert(B == nullptr);
              for (BBNodeT *SubChild : N->successors()) {
                llvm::BasicBlock *SubB = SubChild->getOriginalNode();
                llvm::ConstantInt *IndexConstant = Switch->findCaseDest(SubB);
                revng_assert(IndexConstant != nullptr);
                CaseSet.insert(IndexConstant);
              }
              CaseValuesRegular.push_back(CaseSet);
            } else {
              revng_assert(B != nullptr);

              // We may be in presence of the default case. In this situation
              // place arbitarily the 0 case value.
              llvm::ConstantInt *IndexConstant;
              if (Switch->getDefaultDest() != B) {
                IndexConstant = Switch->findCaseDest(B);
                revng_assert(IndexConstant != nullptr);
                CaseSet.insert(IndexConstant);
                CaseValuesRegular.push_back(CaseSet);
              } else {
                DefaultASTNode = A;
              }
            }
          }
          Index++;
        }
      }

      // Fill the vector of nodes.
      RegularSwitchNode::case_container Cases;
      for (ASTNode *N : ASTChildren) {
        if (N != PostDomASTNode and N != DefaultASTNode) {
          Cases.push_back(N);
        }
      }

      if (Node->isDispatcher()) {
        for (uint64_t Index = 0; Index < Cases.size(); Index++) {
          CaseValuesCheck.push_back(Index);
        }
      }

      // Construct a regular or check switch node depending on the fact that
      // we actually have the condition value.
      if (!Node->isDispatcher()) {
        ASTObject.reset(new RegularSwitchNode(SwitchValue,
                                              std::move(Cases),
                                              std::move(CaseValuesRegular),
                                              DefaultASTNode,
                                              PostDomASTNode));
      } else {
        // Build a SwitchDispatcherNode starting from nodes dispatcher nodes.
        ASTObject.reset(new SwitchDispatcherNode(std::move(Cases),
                                                 std::move(CaseValuesCheck),
                                                 nullptr,
                                                 PostDomASTNode));
      }
    } else {
      switch (Children.size()) {
      case 3: {
        revng_assert(not Node->isBreak() and not Node->isContinue()
                     and not Node->isSet());

        // If we are creating the AST for the check node, create the adequate
        // AST node preserving the then and else branches, otherwise create a
        // classical node.
        // Create the conditional expression associated with the if node.
        using UniqueExpr = ASTTree::expr_unique_ptr;
        using ExprDestruct = ASTTree::expr_destructor;
        auto *OriginalNode = Node->getOriginalNode();
        UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
        ExprNode *CondExprNode = AST.addCondExpr(std::move(CondExpr));
        ASTObject.reset(new IfNode(Node,
                                   CondExprNode,
                                   ASTChildren[0],
                                   ASTChildren[2],
                                   ASTChildren[1]));
      } break;
      case 2: {
        revng_assert(not Node->isBreak() and not Node->isContinue()
                     and not Node->isSet());

        // If we are creating the AST for the switch tree, create the adequate,
        // AST node, otherwise create a classical node.
        // Create the conditional expression associated with the if node.
        using UniqueExpr = ASTTree::expr_unique_ptr;
        using ExprDestruct = ASTTree::expr_destructor;
        auto *OriginalNode = Node->getOriginalNode();
        UniqueExpr CondExpr(new AtomicNode(OriginalNode), ExprDestruct());
        ExprNode *CondExprNode = AST.addCondExpr(std::move(CondExpr));
        ASTObject.reset(new IfNode(Node,
                                   CondExprNode,
                                   ASTChildren[0],
                                   ASTChildren[1],
                                   nullptr));
      } break;
      case 1: {
        revng_assert(not Node->isBreak() and not Node->isContinue());
        if (Node->isSet()) {
          ASTObject.reset(new SetNode(Node, ASTChildren[0]));
        } else {
          ASTObject.reset(new CodeNode(Node, ASTChildren[0]));
        }
      } break;
      case 0: {
        if (Node->isBreak())
          ASTObject.reset(new BreakNode());
        else if (Node->isContinue())
          ASTObject.reset(new ContinueNode());
        else if (Node->isSet())
          ASTObject.reset(new SetNode(Node));
        else if (Node->isEmpty() or Node->isCode())
          ASTObject.reset(new CodeNode(Node, nullptr));
        else
          revng_abort();
      } break;
      }
    }
    AST.addASTNode(Node, std::move(ASTObject));
  }

  // Set in the ASTTree object the root node.
  BasicBlockNode<NodeT> *Root = ASTDT.getRootNode()->getBlock();
  ASTNode *RootNode = AST.findASTNode(Root);

  // Serialize the graph starting from the root node.
  CombLogger << "Serializing first AST draft:\n";
  AST.setRoot(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "First-draft");
  }

  // Create sequence nodes.
  CombLogger << "Performing sequence insertion:\n";
  RootNode = createSequence(AST, RootNode);
  AST.setRoot(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-sequence");
  }

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless dummies simplification:\n";
  simplifyDummies(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-dummies-removal");
  }

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless sequence simplification:\n";
  RootNode = simplifyAtomicSequence(RootNode);
  AST.setRoot(RootNode);
  if (CombLogger.isEnabled()) {
    AST.dumpOnFile("ast", FunctionName, "After-sequence-simplification");
  }

  // Remove danling nodes (possibly created by the de-optimization pass, after
  // disconnecting the first CFG node corresponding to the simplified AST node),
  // and superfluos dummy nodes
  removeNotReachables();
  purgeTrivialDummies();
}

// Get reference to the AST object which is inside the RegionCFG object
template<class NodeT>
inline ASTTree &RegionCFG<NodeT>::getAST() {
  return AST;
}

template<class NodeT>
inline void RegionCFG<NodeT>::removeNotReachables() {

  // Remove nodes that have no predecessors (nodes that are the result of node
  // cloning and that remains dandling around).
  bool Difference = true;
  while (Difference) {
    Difference = false;
    BasicBlockNode<NodeT> *Entry = &getEntryNode();
    for (auto It = begin(); It != end(); It++) {
      if ((Entry != *It and (*It)->predecessor_size() == 0)) {

        removeNode(*It);
        Difference = true;
        break;
      }
    }
  }
}

template<class NodeT>
inline void
RegionCFG<NodeT>::removeNotReachables(std::vector<MetaRegion<NodeT> *> &MS) {

  // Remove nodes that have no predecessors (nodes that are the result of node
  // cloning and that remains dandling around).
  bool Difference = true;
  while (Difference) {
    Difference = false;
    BasicBlockNode<NodeT> *Entry = &getEntryNode();
    for (auto It = begin(); It != end(); It++) {
      if ((Entry != *It and (*It)->predecessor_size() == 0)) {
        for (MetaRegion<NodeT> *M : MS) {
          M->removeNode(*It);
        }
        removeNode(*It);
        Difference = true;
        break;
      }
    }
  }
}

template<class NodeT>
inline bool RegionCFG<NodeT>::isDAG() {
  for (llvm::scc_iterator<RegionCFG<NodeT> *> I = llvm::scc_begin(this),
                                              IE = llvm::scc_end(this);
       I != IE;
       ++I) {
    const std::vector<BasicBlockNode<NodeT> *> &SCC = *I;
    if (SCC.size() != 1) {
      return false;
    } else {
      BasicBlockNode<NodeT> *Node = SCC[0];
      for (BasicBlockNode<NodeT> *Successor : Node->successors()) {
        if (Successor == Node) {
          return false;
        }
      }
    }
  }

  return true;
}

template<class NodeT>
inline bool
RegionCFG<NodeT>::isTopologicallyEquivalent(RegionCFG &Other) const {

  // The algorithm inspects in a depth first fashion the two graphs, and check
  // that they are topologically equivalent. Take care that this function may
  // return true if there are nodes not reachable from the entry node.

  // Early failure if the number of nodes composing the two CFG is different.
  if (size() != Other.size()) {
    return false;
  }

  // Retrieve the entry nodes of the two `RegionCFG` under analysis.
  BasicBlockNode<NodeT> &Entry = getEntryNode();
  BasicBlockNode<NodeT> &OtherEntry = Other.getEntryNode();

  // Check that the only node without predecessors is the entry node.
  for (const BasicBlockNode<NodeT> *Node : nodes()) {
    if (Node != &Entry and Node->predecessor_size() == 0) {
      return false;
    }
  }

  // Check that the only node without predecessors is the entry node.
  for (const BasicBlockNode<NodeT> *Node : Other.nodes()) {
    if (Node != &OtherEntry and Node->predecessor_size() == 0) {
      return false;
    }
  }

  // Call to a `BasicBlockNode` method which does a deep and recursive
  // comparison of a node and its successors.
  return Entry.isEquivalentTo(&OtherEntry);
}

template<class NodeT>
inline void RegionCFG<NodeT>::weave() {

  // Check that we are in a valid state of the graph.
  revng_assert(isDAG());

  // Collect useful objects.
  RegionCFG<NodeT> &Graph = *this;
  BBNodeT *Entry = &getEntryNode();

  BasicBlockNodeTVect ExitNodes;
  for (BBNodeT *Node : Graph) {
    if (Node->successor_size() == 0) {
      ExitNodes.push_back(Node);
    }
  }

  // Add a new virtual sink node to compute the postdominator.
  BasicBlockNode<NodeT> *Sink = Graph.addArtificialNode("Sink");
  for (BasicBlockNode<NodeT> *Exit : ExitNodes) {
    addEdge(EdgeDescriptor(Exit, Sink));
  }

  DT.recalculate(Graph);
  PDT.recalculate(Graph);

  // Iterate over all the nodes in post order.
  for (BBNodeT *POTBB : post_order(Entry)) {

    // If we find a switch node we can start the weaving analysis.
    if (POTBB->successor_size() > 2) {
      BBNodeT *Switch = POTBB;
      if (CombLogger.isEnabled()) {
        CombLogger << "Looking at switch node: " << Switch->getName() << "\n";
        dumpDotOnFile("weaves",
                      FunctionName,
                      "Weaving-region-" + RegionName + "-debug");
      }

      // Collect the case nodes of the switch.
      BasicBlockNodeTSet CaseSet;
      for (BBNodeT *Successor : Switch->successors()) {
        CaseSet.insert(Successor);
      }

      // Find the postdominator of the switch.
      BBNodeT *PostDom = PDT[Switch]->getIDom()->getBlock();
      revng_assert(PostDom != nullptr);

      // Iterate over all the nodes "in the body" of the switch in reverse post
      // order.
      llvm::SmallPtrSet<BBNodeT *, 1> PostDomSet;
      PostDomSet.insert(PostDom);
      ReversePostOrderTraversalExt RPOT(Switch, PostDomSet);

      revng_log(CombLogger,
                "Dumping the candidates that may initiate weaving:");

      for (BBNodeT *RPOTBB : RPOT) {
        // Skip the switch and its post-dominator
        if (RPOTBB == Switch or RPOTBB == PostDom)
          continue;

        // Do not attempt weaving for case nodes.
        if (CaseSet.count(RPOTBB) != 0)
          continue;

        if (CombLogger.isEnabled()) {
          CombLogger << RPOTBB->getName() << "\n";
        }

        BasicBlockNodeTVect PostDominatedCases;
        for (BBNodeT *Case : CaseSet)
          if (PDT.dominates(RPOTBB, Case))
            PostDominatedCases.push_back(Case);

        // Criterion to check if we need to perform the weaving. Specifically,
        // we need to perform a weaving if we find a node (between the switch
        // and its postdominator) that postdominates more than 1 of the cases.
        // Note: it cannot postdominate not all of them, otherwise it would be
        // the immediate postdominator of the switch, that we have explicitly
        // excluded.
        auto NumPostDominatedCases = PostDominatedCases.size();
        revng_assert(NumPostDominatedCases != CaseSet.size());
        if (NumPostDominatedCases > 1U) {

          // Create the new sub-switch node.
          BasicBlockNodeT *NewSwitch = nullptr;
          if (Switch->isDispatcher()) {
            NewSwitch = addDispatcher(Switch->getNameStr() + " weaved");
          } else if (Switch->isCode()) {
            NewSwitch = addNode(Switch->getOriginalNode(),
                                Switch->getNameStr() + " weaved");
          } else {
            revng_unreachable("unexpected switch");
          }
          revng_assert(not NewSwitch);
          revng_assert(not NewSwitch->successor_size());
          NewSwitch->setWeaved(true);

          // Connect the old switch to the new one.
          addEdge(EdgeDescriptor(Switch, NewSwitch));

          // Incremental update of DT and PDT.
          DT.insertEdge(Switch, NewSwitch);
          PDT.insertEdge(Switch, NewSwitch);

          // Iterate over all the case nodes that we found, moving all the
          // necessary edges and update of DT and PDT.
          for (BasicBlockNodeT *Case : PostDominatedCases) {

            removeEdge(EdgeDescriptor(Switch, Case));
            DT.deleteEdge(Switch, Case);
            PDT.deleteEdge(Switch, Case);

            addEdge(EdgeDescriptor(NewSwitch, Case));
            DT.insertEdge(NewSwitch, Case);
            PDT.insertEdge(NewSwitch, Case);
          }

          for (BBNodeT *N : PostDominatedCases)
            CaseSet.erase(N);
          CaseSet.insert(NewSwitch);
        }
      }
    }
  }

  // Purge the final sink used for computing the postdominator tree.
  purgeVirtualSink(Sink);
}

#endif // REVNGC_RESTRUCTURE_CFG_REGIONCFGTREEIMPL_H
