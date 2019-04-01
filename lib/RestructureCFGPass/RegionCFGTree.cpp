/// \file RegionCFGTree.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <fstream>
#include <sys/stat.h>

// LLVM includes
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"
#include <llvm/IR/Instructions.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// Local includes
#include "MetaRegion.h"

// EdgeDescriptor is a handy way to create and manipulate edges on the
// RegionCFG.
using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

// BBNodeToBBMap is a map that contains the original link to the LLVM basic
// block.
using BBNodeToBBMap = std::map<BasicBlockNode *, llvm::BasicBlock *>;

// Helper function that visit an AST tree and creates the sequence nodes
static ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
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
    } else if (auto *Code = llvm::dyn_cast<CodeNode>(Node)) {
      // TODO: confirm that doesn't make sense to process a code node.
    } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {
      // TODO: confirm that this phase is not needed since the processing is
      //       done inside the processing of each SCS region.
    }
  }

  return RootSequenceNode;
}

// Helper function that simplifies useless dummy nodes
static void simplifyDummies(ASTNode *RootNode) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
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

  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      simplifyDummies(If->getThen());
    }
    if (If->hasElse()) {
      simplifyDummies(If->getElse());
    }
  }
}

// Helper function which simplifies sequence nodes composed by a single AST
// node.
ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    if (Sequence->listSize() == 0) {
      RootNode = nullptr;
    } else if (Sequence->listSize() == 1) {
      RootNode = Sequence->getNodeN(0);
      RootNode = simplifyAtomicSequence(RootNode);
    } else {
      for (ASTNode *Node : Sequence->nodes()) {
        Node = simplifyAtomicSequence(Node);
      }
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasThen()) {
      If->setThen(simplifyAtomicSequence(If->getThen()));
    }
    if (If->hasElse()) {
      If->setElse(simplifyAtomicSequence(If->getElse()));
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // TODO: check if this is not needed as the simplification is done for each
    //       SCS region.
    // After flattening this situation may arise again.
    if (Scs->getBody())
      Scs->setBody(simplifyAtomicSequence(Scs->getBody()));
  }

  return RootNode;
}

void RegionCFG::setFunctionName(std::string Name) {
  FunctionName = Name;
}

void RegionCFG::setRegionName(std::string Name) {
  RegionName = Name;
}

std::string RegionCFG::getFunctionName() {
  return FunctionName;
}

std::string RegionCFG::getRegionName() {
  return RegionName;
}

BasicBlockNode *RegionCFG::addNode(llvm::StringRef Name) {
  BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(this, Name));
  BasicBlockNode *Result = BlockNodes.back().get();
  revng_log(CombLogger,
            "Building " << Name << " at address: " << Result << "\n");
  return Result;
}

BasicBlockNode *RegionCFG::cloneNode(const BasicBlockNode &OriginalNode) {
  BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(OriginalNode, this));
  BasicBlockNode *New = BlockNodes.back().get();
  // TODO: find a way to append to the original name the "cloned" suffix. Simply
  //       concatenating with as below causes memory corruption (StringRef).
  //New->setName(OriginalNode.getName() + " cloned");
  return New;
}

void RegionCFG::removeNode(BasicBlockNode *Node) {

  CombLogger << "Removing node named: " << Node->getNameStr() << "\n";

  for (BasicBlockNode *Predecessor : Node->predecessors()) {
    Predecessor->removeSuccessor(Node);
  }

  for (BasicBlockNode *Successor : Node->successors()) {
    Successor->removePredecessor(Node);
  }

  for (auto It = BlockNodes.begin(); It != BlockNodes.end(); It++) {
    if ((*It).get() == Node) {
      BlockNodes.erase(It);
      break;
    }
  }
}

static void copyNeighbors(BasicBlockNode *Dst, BasicBlockNode *Src) {
  for (BasicBlockNode *Succ : Src->successors())
    Dst->addSuccessor(Succ);
  for (BasicBlockNode *Pred : Src->predecessors())
    Dst->addPredecessor(Pred);
}

void RegionCFG::insertBulkNodes(std::set<BasicBlockNode *> &Nodes,
                                BasicBlockNode *Head,
                                RegionCFG::BBNodeMap &SubstitutionMap) {
  revng_assert(BlockNodes.empty());

  for (BasicBlockNode *Node : Nodes) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(*Node, this));
    BasicBlockNode *New = BlockNodes.back().get();
    SubstitutionMap[Node] = New;
    // The copy constructor used above does not bring along the successors and
    // the predecessors, neither adjusts the parent.
    // The following lines are a hack to fix this problem, but they momentarily
    // build a broken data structure where the predecessors and the successors
    // of the New BasicBlockNodes in *this still refer to the BasicBlockNodes in
    // the Parent CFGRegion of Nodes. This will be fixed later by updatePointers
    copyNeighbors(New, Node);
  }

  revng_assert(Head != nullptr);
  EntryNode = SubstitutionMap[Head];
  revng_assert(EntryNode != nullptr);
  // Fix the hack above
  for (std::unique_ptr<BasicBlockNode> &Node : BlockNodes)
    Node->updatePointers(SubstitutionMap);
}

llvm::iterator_range<RegionCFG::links_container::iterator>
RegionCFG::copyNodesAndEdgesFrom(RegionCFG *O, BBNodeMap &SubstitutionMap) {
  size_t NumCurrNodes = size();

  for (BasicBlockNode *Node : *O) {
    BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(*Node, this));
    BasicBlockNode *New = BlockNodes.back().get();
    SubstitutionMap[Node] = New;
    copyNeighbors(New, Node);
  }

  links_container::iterator BeginInserted = BlockNodes.begin() + NumCurrNodes;
  links_container::iterator EndInserted = BlockNodes.end();
  using MovedIteratorRange = llvm::iterator_range<links_container::iterator>;
  MovedIteratorRange Result = llvm::make_range(BeginInserted, EndInserted);
  for (std::unique_ptr<BasicBlockNode> &NewNode : Result)
    NewNode->updatePointers(SubstitutionMap);
  return Result;
}

void RegionCFG::connectBreakNode(std::set<EdgeDescriptor> &Outgoing,
                                 const BBNodeMap &SubstitutionMap) {
  for (EdgeDescriptor Edge : Outgoing) {

    // Create a new break for each outgoing edge.
    BasicBlockNode *Break = addBreak();
    if (not Edge.first->isCheck()) {
      addEdge(EdgeDescriptor(SubstitutionMap.at(Edge.first), Break));
    } else {
      revng_assert(Edge.second == Edge.first->getTrue()
                   or Edge.second == Edge.first->getFalse());
      if (Edge.second == Edge.first->getTrue())
        SubstitutionMap.at(Edge.first)->setTrue(Break);
      else
        SubstitutionMap.at(Edge.first)->setFalse(Break);
    }
  }
}

void RegionCFG::connectContinueNode() {
  std::vector<BasicBlockNode *> ContinueNodes;

  // We need to pre-save the edges to avoid breaking the predecessor iterator
  for (BasicBlockNode *Source : EntryNode->predecessors()) {
    ContinueNodes.push_back(Source);
  }
  for (BasicBlockNode *Source : ContinueNodes) {

    // Create a new continue node for each retreating edge.
    BasicBlockNode *Continue = addContinue();
    moveEdgeTarget(EdgeDescriptor(Source, EntryNode), Continue);
  }
}

std::vector<BasicBlockNode *>
RegionCFG::orderNodes(std::vector<BasicBlockNode *> &L, bool DoReverse) {
  std::set<BasicBlockNode *> ToOrder;
  ToOrder.insert(L.begin(), L.end());
  llvm::ReversePostOrderTraversal<BasicBlockNode *> RPOT(EntryNode);
  std::vector<BasicBlockNode *> Result;

  if (DoReverse) {
    std::reverse(RPOT.begin(), RPOT.end());
  }

  for (BasicBlockNode *RPOTBB : RPOT) {
    if (ToOrder.count(RPOTBB) != 0) {
      Result.push_back(RPOTBB);
    }
  }

  revng_assert(L.size() == Result.size());

  return Result;
}

template<typename StreamT>
void RegionCFG::streamNode(StreamT &S, const BasicBlockNode *BB) const {
  unsigned NodeID = BB->getID();
  S << "\"" << NodeID << "\"";
  S << " ["
    << "label=\"ID: " << NodeID << " Name: " << BB->getName().str() << "\"";
  if (BB == EntryNode)
    S << ",fillcolor=green,style=filled";
  S << "];\n";
}

/// \brief Dump a GraphViz file on stdout representing this function
template<typename StreamT>
void RegionCFG::dumpDot(StreamT &S) const {
  S << "digraph CFGFunction {\n";

  for (const std::unique_ptr<BasicBlockNode> &BB : BlockNodes) {
    streamNode(S, BB.get());
    for (auto &Successor : BB->successors()) {
      unsigned PredID = BB->getID();
      unsigned SuccID = Successor->getID();
      S << "\"" << PredID << "\""
        << " -> \"" << SuccID << "\"";
      if (BB->isCheck() and BB->getFalse() == Successor)
        S << " [color=red];\n";
      else
        S << " [color=green];\n";
    }
  }
  S << "}\n";
}

void RegionCFG::dumpDotOnFile(std::string FolderName,
                              std::string FunctionName,
                              std::string FileName) const {
  std::ofstream DotFile;
  std::string PathName = FolderName + "/" + FunctionName;
  mkdir(FolderName.c_str(), 0775);
  mkdir(PathName.c_str(), 0775);
  DotFile.open(PathName + "/" + FileName + ".dot");
  dumpDot(DotFile);
}

void RegionCFG::purgeDummies() {
  RegionCFG &Graph = *this;
  bool AnotherIteration = true;

  while (AnotherIteration) {
    AnotherIteration = false;

    for (auto It = Graph.begin(); It != Graph.end(); It++) {
      if (((*It)->isEmpty()) and ((*It)->predecessor_size() == 1)
          and ((*It)->successor_size() == 1)) {

        if (CombLogger.isEnabled()) {
          CombLogger << "Purging dummy node " << (*It)->getNameStr() << "\n";
        }

        BasicBlockNode *Predecessor = (*It)->getPredecessorI(0);
        BasicBlockNode *Successor = (*It)->getSuccessorI(0);

        // Connect directly predecessor and successor, and remove the dummy node
        // under analysis
        moveEdgeTarget({ Predecessor, *It }, Successor);
        DT.insertEdge(Predecessor, Successor);
        PDT.insertEdge(Predecessor, Successor);

        DT.eraseNode(*It);
        PDT.eraseNode(*It);
        Graph.removeNode(*It);

        AnotherIteration = true;
        break;
      }
    }
  }
}

void RegionCFG::purgeVirtualSink(BasicBlockNode *Sink) {

  RegionCFG &Graph = *this;

  std::vector<BasicBlockNode *> WorkList;
  std::vector<BasicBlockNode *> PurgeList;

  WorkList.push_back(Sink);

  while (!WorkList.empty()) {
    BasicBlockNode *CurrentNode = WorkList.back();
    WorkList.pop_back();

    if (CurrentNode->isEmpty()) {
      PurgeList.push_back(CurrentNode);

      for (BasicBlockNode *Predecessor : CurrentNode->predecessors()) {
        WorkList.push_back(Predecessor);
      }
    }
  }

  for (BasicBlockNode *Purge : PurgeList) {
    Graph.removeNode(Purge);
  }
}

std::vector<BasicBlockNode *>
RegionCFG::getInterestingNodes(BasicBlockNode *Cond) {

  RegionCFG &Graph = *this;

  // Retrieve the immediate postdominator.
  llvm::DomTreeNodeBase<BasicBlockNode> *PostBase = PDT[Cond]->getIDom();
  BasicBlockNode *PostDominator = PostBase->getBlock();

  std::set<BasicBlockNode *> Candidates = findReachableNodes(*Cond,
                                                             *PostDominator);

  std::vector<BasicBlockNode *> NotDominatedCandidates;
  for (BasicBlockNode *Node : Candidates) {
    if (!DT.dominates(Cond, Node)) {
      NotDominatedCandidates.push_back(Node);
    }
  }

  // TODO: Check that this is the order that we want.
  NotDominatedCandidates = Graph.orderNodes(NotDominatedCandidates, true);

  return NotDominatedCandidates;
}

void RegionCFG::inflate() {

   revng_assert(isDAG());

  // Apply the comb to a RegionCFG object.
  // TODO: handle all the collapsed regions.
  RegionCFG &Graph = *this;

  // Refresh information of dominator and postdominator trees.
  DT.recalculate(Graph);
  PDT.recalculate(Graph);

  // Collect entry and exit nodes.
  BasicBlockNode *EntryNode = &Graph.getEntryNode();
  std::vector<BasicBlockNode *> ExitNodes;
  for (auto It = Graph.begin(); It != Graph.end(); It++) {
    if ((*It)->successor_size() == 0) {
      ExitNodes.push_back(*It);
    }
  }

  if (CombLogger.isEnabled()) {
    CombLogger << "The entry node is:\n";
    CombLogger << EntryNode->getNameStr() << "\n";
    CombLogger << "In the graph the exit nodes are:\n";
    for (BasicBlockNode *Node : ExitNodes) {
      CombLogger << Node->getNameStr() << "\n";
    }
  }

  // Helper data structure for exit reachability computation.
  std::set<BasicBlockNode *> ConditionalBlacklist;
  std::map<BasicBlockNode *, std::set<BasicBlockNode *>> ReachableExits;

  // Collect nodes reachable from each exit node in the graph.
  for (BasicBlockNode *Exit : ExitNodes) {
    CombLogger << "From exit node: " << Exit->getNameStr() << "\n";
    CombLogger << "We can reach:\n";
    for (BasicBlockNode *Node : llvm::inverse_depth_first(Exit)) {
      CombLogger << Node->getNameStr() << "\n";
      ReachableExits[Node].insert(Exit);
    }
  }

  // Dump graph before virtual sink add.
  if (CombLogger.isEnabled()) {
    CombLogger << "Graph before sink addition is:\n";
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-before-sink");
  }

  // Add a new virtual sink node to which all the exit nodes are connected.
  BasicBlockNode *Sink = Graph.addArtificialNode("Virtual sink");
  for (BasicBlockNode *Exit : ExitNodes) {
    addEdge(EdgeDescriptor(Exit, Sink));
  }

  // Dump graph after virtual sink add.
  if (CombLogger.isEnabled()) {
    CombLogger << "Graph after sink addition is:\n";
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-after-sink");
  }

  // Collect all the conditional nodes in the graph.
  // This is the working list of conditional nodes on which we will operate and
  // will contain only the filtered conditionals.
  std::vector<BasicBlockNode *> ConditionalNodes;

  // This set contains all the conditional nodes present in the graph
  std::set<BasicBlockNode *> ConditionalNodesComplete;

  for (auto It = Graph.begin(); It != Graph.end(); It++) {
    revng_assert((*It)->successor_size() < 3);
    if ((*It)->successor_size() == 2) {

      // Check that the intersection of exits nodes reachable from the then and
      // else branches are disjoint.
      std::set<BasicBlockNode *>
        ThenExits = ReachableExits[(*It)->getSuccessorI(0)];
      std::set<BasicBlockNode *>
        ElseExits = ReachableExits[(*It)->getSuccessorI(1)];
      std::vector<BasicBlockNode *> Intersection;
      std::set_intersection(ThenExits.begin(),
                            ThenExits.end(),
                            ElseExits.begin(),
                            ElseExits.end(),
                            std::back_inserter(Intersection));

      // Check that we do not dominate at maximum on of the two sets of
      // reachable exits.
      bool ThenIsDominated = true;
      bool ElseIsDominated = true;
      for (BasicBlockNode *Exit : ThenExits) {
        if (not DT.dominates(*It, Exit)) {
          ThenIsDominated = false;
        }
      }
      for (BasicBlockNode *Exit : ElseExits) {
        if (not DT.dominates(*It, Exit)) {
          ElseIsDominated = false;
        }
      }

      // This check adds a conditional nodes if the sets of reachable exits are
      // not disjoint or if we do not dominate both the reachable exit sets
      // (note that we may not dominate one of the two reachable sets, meaning
      // the fallthrough branch, but we need to dominate the other in such a way
      // that we can completely absorb it).
      if (Intersection.size() != 0
          or (not (ThenIsDominated or ElseIsDominated))) {
        ConditionalNodes.push_back(*It);
        ConditionalNodesComplete.insert(*It);
      } else {
        CombLogger << "Blacklisted conditional: " << (*It)->getNameStr()
                   << "\n";
      }
    }
  }

  // TODO: reverse this order, with std::vector I can only pop_back.
  ConditionalNodes = Graph.orderNodes(ConditionalNodes, false);

  if (CombLogger.isEnabled()) {
    CombLogger << "Conditional nodes present in the graph are:\n";
    for (BasicBlockNode *Node : ConditionalNodes) {
      CombLogger << Node->getNameStr() << "\n";
    }
  }

  // Refresh information of dominator and postdominator trees.
  DT.recalculate(Graph);
  PDT.recalculate(Graph);

  while (!ConditionalNodes.empty()) {

    // Process each conditional node after ordering it.
    BasicBlockNode *Conditional = ConditionalNodes.back();
    ConditionalNodes.pop_back();
    if (CombLogger.isEnabled()) {
      CombLogger << "Analyzing conditional node " << Conditional->getNameStr()
                 << "\n";
    }
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-conditional-"
                          + Conditional->getNameStr() + "-begin");
    CombLogger.emit();

    // Get all the nodes reachable from the current conditional node (stopping
    // at the immediate postdominator) and that we want to duplicate/split.
    std::vector<BasicBlockNode *> NotDominatedCandidates;
    NotDominatedCandidates = getInterestingNodes(Conditional);

    int Iteration = 0;
    while (!NotDominatedCandidates.empty()) {

      // TODO: Remove this
      // NotDominatedCandidates = getInterestingNodes(Conditional);

      if (CombLogger.isEnabled()) {
        CombLogger << "Analyzing candidate nodes\n ";
      }
      BasicBlockNode *Candidate = NotDominatedCandidates.back();
      NotDominatedCandidates.pop_back();
      if (CombLogger.isEnabled()) {
        CombLogger << "Analyzing candidate " << Candidate->getNameStr() << "\n";
      }

      // Decide wether to insert a dummy or to duplicate.
      if (Candidate->predecessor_size() > 2) {

        // Insert a dummy node.
        if (CombLogger.isEnabled()) {
          CombLogger << "Inserting a dummy node for ";
          CombLogger << Candidate->getNameStr() << "\n";
        }

        typedef enum { Left, Right } Side;

        std::vector<Side> Sides{ Left, Right };
        std::map<Side, BasicBlockNode *> Dummies;

        for (Side S : Sides) {
          BasicBlockNode *Dummy = Graph.addArtificialNode("dummy");
          Dummies[S] = Dummy;
        }

        std::vector<BasicBlockNode *> Predecessors;

        CombLogger << "Current predecessors are:\n";
        for (BasicBlockNode *Predecessor : Candidate->predecessors()) {
          CombLogger << Predecessor->getNameStr() << "\n";
          Predecessors.push_back(Predecessor);
        }

        for (BasicBlockNode *Predecessor : Predecessors) {
          if (CombLogger.isEnabled()) {
            CombLogger << "Moving edge from predecessor ";
            CombLogger << Predecessor->getNameStr() << "\n";
          }
          if (DT.dominates(Conditional, Predecessor)) {
            moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate),
                           Dummies[Left]);

            // Inform the dominator and postdominator tree about the update
            DT.insertEdge(Predecessor, Dummies[Left]);
            PDT.insertEdge(Predecessor, Dummies[Left]);
            DT.deleteEdge(Predecessor, Candidate);
            PDT.deleteEdge(Predecessor, Candidate);
          } else {
            moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate),
                           Dummies[Right]);

            // Inform the dominator and postdominator tree about the update
            DT.insertEdge(Predecessor, Dummies[Right]);
            PDT.insertEdge(Predecessor, Dummies[Right]);
            DT.deleteEdge(Predecessor, Candidate);
            PDT.deleteEdge(Predecessor, Candidate);
          }
        }

        for (Side S : Sides) {
          addEdge(EdgeDescriptor(Dummies[S], Candidate));

          // Inform the dominator and postdominator tree about the update
          DT.insertEdge(Dummies[S], Candidate);
          PDT.insertEdge(Dummies[S], Candidate);
        }
      } else {

        // Duplicate node.
        if (CombLogger.isEnabled()) {
          CombLogger << "Duplicating node for ";
          CombLogger << Candidate->getNameStr() << "\n";
        }

        BasicBlockNode *Duplicated = Graph.cloneNode(*Candidate);
        revng_assert(Duplicated != nullptr);

        // If the node we are duplicating is a conditional node, add it to the
        // working list of the conditional nodes.
        if (ConditionalNodesComplete.count(Candidate) != 0) {
          ConditionalNodes.push_back(Duplicated);
          ConditionalNodesComplete.insert(Duplicated);
        }

        // Specifically handle the check idx node situation.
        if (Candidate->isCheck()) {
          revng_assert(Candidate->getTrue() != nullptr
                       and Candidate->getFalse() != nullptr);
          BasicBlockNode *TrueSuccessor = Candidate->getTrue();
          BasicBlockNode *FalseSuccessor = Candidate->getFalse();
          Duplicated->setTrue(TrueSuccessor);
          DT.insertEdge(Duplicated, TrueSuccessor);
          Duplicated->setFalse(FalseSuccessor);
          DT.insertEdge(Duplicated, FalseSuccessor);

        } else {
          for (BasicBlockNode *Successor : Candidate->successors()) {
            addEdge(EdgeDescriptor(Duplicated, Successor));

            // Inform the dominator and postdominator tree about the update
            DT.insertEdge(Duplicated, Successor);
            PDT.insertEdge(Duplicated, Successor);
          }
        }
        std::vector<BasicBlockNode *> Predecessors;

        for (BasicBlockNode *Predecessor : Candidate->predecessors()) {
          Predecessors.push_back(Predecessor);
        }

        for (BasicBlockNode *Predecessor : Predecessors) {
          if (!DT.dominates(Conditional, Predecessor)) {
            moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate), Duplicated);

            // Inform the dominator and postdominator tree about the update
            DT.insertEdge(Predecessor, Duplicated);
            PDT.insertEdge(Predecessor, Duplicated);
            DT.deleteEdge(Predecessor, Candidate);
            PDT.deleteEdge(Predecessor, Candidate);
          }
        }
      }

      // Purge extra dummies at each iteration
      purgeDummies();

      if (CombLogger.isEnabled()) {
        Graph.dumpDotOnFile("inflates",
                            FunctionName,
                            "Region-" + RegionName + "-conditional-"
                              + Conditional->getNameStr() + "-"
                              + std::to_string(Iteration));
      }
      Iteration++;

      CombLogger << "Finished looking at: ";
      CombLogger << Conditional->getNameStr() << "\n";

      // Refresh the info on candidates.
      NotDominatedCandidates = getInterestingNodes(Conditional);
    }
  }

  // Purge extra dummy nodes introduced.
  purgeDummies();
  purgeVirtualSink(Sink);

  // if (CombLogger.isEnabled()) {
  CombLogger << "Graph after combing is:\n";
  Graph.dumpDotOnFile("inflates",
                      FunctionName,
                      "Region-" + RegionName + "-after-combing");
  //}
}

void RegionCFG::generateAst(BBNodeToBBMap &OriginalBB) {

  RegionCFG &Graph = *this;

  // Apply combing to the current RegionCFG.
  dumpDotOnFile("dots", FunctionName, "PRECOMB");
  if (ToInflate) {
    CombLogger << "Inflating region " + RegionName + "\n";
    Graph.inflate();
    ToInflate = false;
  }
  dumpDotOnFile("dots", FunctionName, "POSTCOMB");

  // TODO: factorize out the AST generation phase.
  llvm::DominatorTreeBase<BasicBlockNode, false> ASTDT;
  ASTDT.recalculate(Graph);
  ASTDT.updateDFSNumbers();

  CombLogger.emit();

  std::map<int, BasicBlockNode *> DFSNodeMap;

  // Compute the ideal order of visit for creating AST nodes.
  for (BasicBlockNode *Node : Graph.nodes()) {
    DFSNodeMap[ASTDT[Node]->getDFSNumOut()] = Node;
  }

  // Visiting order of the dominator tree.
  if (CombLogger.isEnabled()) {
    for (auto &Pair : DFSNodeMap) {
      CombLogger << Pair.second->getNameStr() << "\n";
    }
  }

  for (auto &Pair : DFSNodeMap) {
    BasicBlockNode *Node = Pair.second;

    // Collect the children nodes in the dominator tree.
    std::vector<llvm::DomTreeNodeBase<BasicBlockNode> *>
      Children = ASTDT[Node]->getChildren();

    std::vector<ASTNode *> ASTChildren;
    std::vector<BasicBlockNode *> BBChildren;
    for (llvm::DomTreeNodeBase<BasicBlockNode> *TreeNode : Children) {
      BasicBlockNode *BlockNode = TreeNode->getBlock();
      ASTNode *ASTPointer = AST.findASTNode(BlockNode);
      ASTChildren.push_back(ASTPointer);
      BBChildren.push_back(BlockNode);
    }

    // Check that the two vector have the same size.
    revng_assert(Children.size() == ASTChildren.size());

    // Handle collapsded node.
    if (Node->isCollapsed()) {
      revng_assert(ASTChildren.size() <= 1);
      if (ASTChildren.size() == 1) {
        RegionCFG *BodyGraph = Node->getCollapsedCFG();
        revng_assert(BodyGraph != nullptr);
        CombLogger << "Inspecting collapsed node: " << Node->getNameStr()
                   << "\n";
        CombLogger.emit();
        BodyGraph->generateAst(OriginalBB);
        ASTNode *Body = BodyGraph->getAST().getRoot();
        std::unique_ptr<ASTNode> ASTObject(new ScsNode(Node,
                                                       Body,
                                                       ASTChildren[0]));
        AST.addASTNode(Node, std::move(ASTObject));
      } else {
        RegionCFG *BodyGraph = Node->getCollapsedCFG();
        CombLogger << "Inspecting collapsed node: " << Node->getNameStr()
                   << "\n";
        CombLogger.emit();
        BodyGraph->generateAst(OriginalBB);
        ASTNode *Body = BodyGraph->getAST().getRoot();
        std::unique_ptr<ASTNode> ASTObject(new ScsNode(Node, Body));
        AST.addASTNode(Node, std::move(ASTObject));
      }
    } else {
      revng_assert(Children.size() < 4);
      std::unique_ptr<ASTNode> ASTObject;
      if (Children.size() == 3) {
        revng_assert(not Node->isBreak() and not Node->isContinue()
                     and not Node->isSet());

        // Create the conditional expression associated with the if node.
        auto CondExpr = std::make_unique<AtomicNode>(Node->getBasicBlock());
        ExprNode *CondExprNode = AST.addCondExpr(std::move(CondExpr));

        // If we are creating the AST for the check node, create the adequate
        // AST node preserving the then and else branches, otherwise create a
        // classical node.
        if (Node->isCheck()) {
          if (BBChildren[0] == Node->getTrue()
              and BBChildren[2] == Node->getFalse()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            ASTChildren[0],
                                            ASTChildren[2],
                                            ASTChildren[1]));
          } else if (BBChildren[2] == Node->getTrue()
                     and BBChildren[0] == Node->getFalse()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            ASTChildren[2],
                                            ASTChildren[0],
                                            ASTChildren[1]));
          } else {
            revng_abort("Then and else branches cannot be matched");
          }
        } else {
          ASTObject.reset(new IfNode(Node,
                                     OriginalBB.at(Node),
                                     CondExprNode,
                                     ASTChildren[0],
                                     ASTChildren[2],
                                     ASTChildren[1]));
        }
      } else if (Children.size() == 2) {
        revng_assert(not Node->isBreak() and not Node->isContinue()
                     and not Node->isSet());

        // Create the conditional expression associated with the if node.
        auto CondExpr = std::make_unique<AtomicNode>(Node->getBasicBlock());
        ExprNode *CondExprNode = AST.addCondExpr(std::move(CondExpr));

        // If we are creating the AST for the switch tree, create the adequate,
        // AST node, otherwise create a classical node.
        if (Node->isCheck()) {
          if (BBChildren[0] == Node->getTrue()
              and BBChildren[1] == Node->getFalse()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            ASTChildren[0],
                                            ASTChildren[1],
                                            nullptr));
          } else if (BBChildren[1] == Node->getTrue()
                     and BBChildren[0] == Node->getFalse()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            ASTChildren[1],
                                            ASTChildren[0],
                                            nullptr));
          } else {
            revng_abort("Then and else branches cannot be matched");
          }
        } else {
          ASTObject.reset(new IfNode(Node,
                                     OriginalBB.at(Node),
                                     CondExprNode,
                                     ASTChildren[0],
                                     ASTChildren[1],
                                     nullptr));
        }
      } else if (Children.size() == 1) {
        revng_assert(not Node->isBreak() and not Node->isContinue());
        if (Node->isSet()) {
          ASTObject.reset(new SetNode(Node, ASTChildren[0]));
        } else if (Node->isCheck()) {

          // We may have a check node with a single then/else branch due to
          // condition blacklisting (the other branch is the fallthrough
          // branch).
          if (BBChildren[0] == Node->getTrue()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            ASTChildren[0],
                                            nullptr,
                                            nullptr));
          } else if (BBChildren[0] == Node->getFalse()) {
            ASTObject.reset(new IfCheckNode(Node,
                                            nullptr,
                                            ASTChildren[0],
                                            nullptr));
          }
        } else {
          ASTObject.reset(new CodeNode(Node, OriginalBB[Node], ASTChildren[0]));
        }
      } else if (Children.size() == 0) {
        if (Node->isBreak())
          ASTObject.reset(new BreakNode());
        else if (Node->isContinue())
          ASTObject.reset(new ContinueNode());
        else if (Node->isSet())
          ASTObject.reset(new SetNode(Node));
        else if (Node->isEmpty() or Node->isCode())
          ASTObject.reset(new CodeNode(Node, OriginalBB[Node], nullptr));
        else
          revng_abort();
      }
      AST.addASTNode(Node, std::move(ASTObject));
    }
  }

  // Set in the ASTTree object the root node.
  BasicBlockNode *Root = ASTDT.getRootNode()->getBlock();
  ASTNode *RootNode = AST.findASTNode(Root);

  // Serialize the graph starting from the root node.
  CombLogger << "Serializing first AST draft:\n";
  AST.setRoot(RootNode);
  AST.dumpOnFile("ast", FunctionName, "First-draft");

  // Create sequence nodes.
  CombLogger << "Performing sequence insertion:\n";
  RootNode = createSequence(AST, RootNode);
  AST.setRoot(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-sequence");

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless dummies simplification:\n";
  simplifyDummies(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-dummies-removal");

  // Simplify useless sequence nodes.
  CombLogger << "Performing useless sequence simplification:\n";
  RootNode = simplifyAtomicSequence(RootNode);
  AST.setRoot(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-sequence-simplification");

  // Remove danling nodes (possibly created by the de-optimization pass, after
  // disconnecting the first CFG node corresponding to the simplified AST node),
  // and superfluos dummy nodes
  removeNotReachables();
  purgeDummies();
}

// Get reference to the AST object which is inside the RegionCFG object
ASTTree &RegionCFG::getAST() {
  return AST;
}

void RegionCFG::removeNotReachables() {

  // Remove nodes that have no predecessors (nodes that are the result of node
  // cloning and that remains dandling around).
  bool Difference = true;
  while (Difference) {
    Difference = false;
    BasicBlockNode *EntryNode = &getEntryNode();
    for (auto It = begin(); It != end(); It++) {
      if ((EntryNode != *It and (*It)->predecessor_size() == 0)) {
        removeNode(*It);
        Difference = true;
        break;
      }
    }
  }
}

void RegionCFG::removeNotReachables(std::vector<MetaRegion *> &MS) {

  // Remove nodes that have no predecessors (nodes that are the result of node
  // cloning and that remains dandling around).
  bool Difference = true;
  while (Difference) {
    Difference = false;
    BasicBlockNode *EntryNode = &getEntryNode();
    for (auto It = begin(); It != end(); It++) {
      if ((EntryNode != *It and (*It)->predecessor_size() == 0)) {
        for (MetaRegion *M : MS) {
          M->removeNode(*It);
        }
        removeNode(*It);
        Difference = true;
        break;
      }
    }
  }
}

bool RegionCFG::isDAG() {
  bool FoundSCC = false;

  for (llvm::scc_iterator<RegionCFG *> I = llvm::scc_begin(this),
                                       IE = llvm::scc_end(this);
                                       I != IE; ++I) {
    const std::vector<BasicBlockNode *> &SCC = *I;
    if (SCC.size() != 1) {
      FoundSCC = true;
    } else {
      BasicBlockNode *Node = SCC[0];
      for (BasicBlockNode *Successor : Node->successors()) {
        if (Successor == Node) {
          FoundSCC = true;
        }
      }
    }
  }

return not FoundSCC;
}
