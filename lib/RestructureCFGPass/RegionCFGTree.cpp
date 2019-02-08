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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTTree.h"
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// EdgeDescriptor is a handy way to create and manipulate edges on the RegionCFG.
using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

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
    }
    #if 0
  } else if (auto *Code = llvm::dyn_cast<CodeNode>(Node)) {
    // TODO: confirm that doesn't make sense to process a code node.
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {
    // TODO: confirm that this phase is not needed since the processing is
    //       done inside the processing of each SCS region.
  }
  #endif
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
static ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {
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
  }
  #if 0
} else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // TODO: check if this is not needed as the simplification is done for each
    //       SCS region.
  }
  #endif

  return RootNode;
}

static void deoptimizeNodes(ASTNode *First, ASTNode *Second) {
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Trying to de-optimize: ";
    CombLogger << First->getName() << " and ";
    CombLogger << Second->getName() << "\n\n";
  }

  BasicBlockNode *FirstEntry = First->getFirstCFG();
  BasicBlockNode *SecondEntry = Second->getFirstCFG();
  assert((FirstEntry != nullptr) and (SecondEntry != nullptr));
  for (BasicBlockNode *Predecessor : FirstEntry->predecessors()) {
    addEdge(EdgeDescriptor(Predecessor, SecondEntry));
  }
  FirstEntry->removeNode();
}

// Helper function to simplify short-circuit IFs
static void simplifyShortCircuit(ASTNode *RootNode) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyShortCircuit(Node);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyShortCircuit(Scs->getBody());
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (If->hasBothBranches()) {

      if (auto InternalIf = llvm::dyn_cast<IfNode>(If->getThen())) {

        // TODO: Refactor this with some kind of iterator
        if (InternalIf->getThen() != nullptr) {
          if (If->getElse()->isEqual(InternalIf->getThen())) {
            if (CombLogger.isEnabled()) {
              CombLogger << "Candidate for short-circuit reduction found:\n";
              CombLogger << "IF " << If->getName() << " and ";
              CombLogger << "IF " << InternalIf->getName() << "\n";
              CombLogger << "Nodes being simplified:\n";
              CombLogger << If->getElse()->getName() << " and ";
              CombLogger << InternalIf->getThen()->getName() << "\n";
            }

            // Deoptimize the CFG
            deoptimizeNodes(If->getElse(), InternalIf->getThen());

            If->setThen(InternalIf->getElse());
            If->setElse(InternalIf->getThen());

            // Absorb the conditional nodes
            If->addConditionalNodesFrom(InternalIf);

            simplifyShortCircuit(If);
          }
        }

        if (InternalIf->getElse() != nullptr) {
          if (If->getElse()->isEqual(InternalIf->getElse())) {
            if (CombLogger.isEnabled()) {
              CombLogger << "Candidate for short-circuit reduction found:\n";
              CombLogger << "IF " << If->getName() << " and ";
              CombLogger << "IF " << InternalIf->getName() << "\n";
              CombLogger << "Nodes being simplified:\n";
              CombLogger << If->getElse()->getName() << " and ";
              CombLogger << InternalIf->getElse()->getName() << "\n";
            }

            // Deoptimize the CFG
            deoptimizeNodes(If->getElse(), InternalIf->getElse());

            If->setThen(InternalIf->getThen());
            If->setElse(InternalIf->getElse());

            // Absorb the conditional nodes
            If->addConditionalNodesFrom(InternalIf);

            simplifyShortCircuit(If);
          }
        }
      }

      if (auto InternalIf = llvm::dyn_cast<IfNode>(If->getElse())) {

        // TODO: Refactor this with some kind of iterator
        if (InternalIf->getThen() != nullptr) {
          if (If->getThen()->isEqual(InternalIf->getThen())) {
            if (CombLogger.isEnabled()) {
              CombLogger << "Candidate for short-circuit reduction found:\n";
              CombLogger << "IF " << If->getName() << " and ";
              CombLogger << "IF " << InternalIf->getName() << "\n";
              CombLogger << "Nodes being simplified:\n";
              CombLogger << If->getThen()->getName() << " and ";
              CombLogger << InternalIf->getThen()->getName() << "\n";
            }

            // Deoptimize the CFG
            deoptimizeNodes(If->getThen(), InternalIf->getThen());

            If->setElse(InternalIf->getElse());
            If->setThen(InternalIf->getThen());

            // Absorb the conditional nodes
            If->addConditionalNodesFrom(InternalIf);

            simplifyShortCircuit(If);
          }
        }

        if (InternalIf->getElse() != nullptr) {
          if (If->getThen()->isEqual(InternalIf->getElse())) {
            if (CombLogger.isEnabled()) {
              CombLogger << "Candidate for short-circuit reduction found:\n";
              CombLogger << "IF " << If->getName() << " and ";
              CombLogger << "IF " << InternalIf->getName() << "\n";
              CombLogger << "Nodes being simplified:\n";
              CombLogger << If->getThen()->getName() << " and ";
              CombLogger << InternalIf->getElse()->getName() << "\n";
            }

            // Deoptimize the CFG
            deoptimizeNodes(If->getThen(), InternalIf->getElse());

            If->setElse(InternalIf->getThen());
            If->setThen(InternalIf->getElse());

            // Absorb the conditional nodes
            If->addConditionalNodesFrom(InternalIf);

            simplifyShortCircuit(If);
          }
        }
      }

    }
  }
}

static void flipEmptyThen(ASTNode *RootNode) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      flipEmptyThen(Node);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (!If->hasThen()) {
      if (CombLogger.isEnabled()) {
        CombLogger << "Flipping then and else branches for : ";
        CombLogger << If->getName() << "\n";
      }
      If->setThen(If->getElse());
      If->setElse(nullptr);
      flipEmptyThen(If->getThen());
    } else {

      // We are sure to have the `then` branch since the previous check did
      // not verify
      flipEmptyThen(If->getThen());

      // We have not the same assurance for the `else` branch
      if (If->hasElse()) {
        flipEmptyThen(If->getElse());
      }
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    flipEmptyThen(Scs->getBody());
  }
}

static void simplifyTrivialShortCircuit(ASTNode *RootNode) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    for (ASTNode *Node : Sequence->nodes()) {
      simplifyTrivialShortCircuit(Node);
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    simplifyTrivialShortCircuit(Scs->getBody());
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    if (!If->hasElse()) {
      if (auto *InternalIf = llvm::dyn_cast<IfNode>(If->getThen())) {
        if (!InternalIf->hasElse()) {
          if (CombLogger.isEnabled()) {
            CombLogger << "Candidate for trivial short-circuit reduction";
            CombLogger << "found:\n";
            CombLogger << "IF " << If->getName() << " and ";
            CombLogger << "If " << InternalIf->getName() << "\n";
            CombLogger << "Nodes being simplified:\n";
            CombLogger << If->getThen()->getName() << " and ";
            CombLogger << InternalIf->getThen()->getName() << "\n";
          }

          If->setThen(InternalIf->getThen());

          // Absorb the conditional nodes
          If->addConditionalNodesFrom(InternalIf);

          simplifyTrivialShortCircuit(RootNode);
        }
      }
    }

    if (If->hasThen()) {
      simplifyTrivialShortCircuit(If->getThen());
    }
    if (If->hasElse()) {
      simplifyTrivialShortCircuit(If->getElse());
    }
  }
}

void RegionCFG::initialize(llvm::Function &F) {

  // Create a new node for each basic block in the module.
  for (llvm::BasicBlock &BB : F) {
    addNode(&BB);
  }

  // Set entry node references.
  Entry = &(F.getEntryBlock());
  EntryNode = &(get(Entry));

  // Connect each node to its successors.
  for (llvm::BasicBlock &BB : F) {
    BasicBlockNode &Node = get(&BB);

    llvm::TerminatorInst *Terminator = BB.getTerminator();
    int SuccessorNumber = Terminator->getNumSuccessors();

    if (SuccessorNumber < 3) {
      // Add the successors to the node.
      for (llvm::BasicBlock *Successor : Terminator->successors()) {
        BasicBlockNode &SuccessorNode = get(Successor);
        Node.addSuccessor(&SuccessorNode);
        SuccessorNode.addPredecessor(&Node);
      }
    } else {

      // HACK: handle switches as a nested tree of ifs.
      std::vector<llvm::BasicBlock *> WorkList;
      for (llvm::BasicBlock *Successor : reverse(Terminator->successors())) {
        WorkList.push_back(Successor);
      }

      BasicBlockNode *PrevDummy = &get(&BB);

      // For each iteration except the last create a new dummy node
      // connecting the successors.
      while (WorkList.size() > 2) {
        BasicBlockNode *NewDummy = addArtificialNode("switch dummy");
        BasicBlockNode *Dest1 = &get(WorkList.back());
        WorkList.pop_back();
        addEdge(EdgeDescriptor(PrevDummy, Dest1));
        addEdge(EdgeDescriptor(PrevDummy, NewDummy));
        PrevDummy = NewDummy;
      }

      BasicBlockNode *Dest1 = &get(WorkList.back());
      WorkList.pop_back();
      BasicBlockNode *Dest2 = &get(WorkList.back());
      WorkList.pop_back();
      revng_assert(WorkList.empty());
      addEdge(EdgeDescriptor(PrevDummy, Dest1));
      addEdge(EdgeDescriptor(PrevDummy, Dest2));
    }
  }
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

BasicBlockNode *RegionCFG::addNode(llvm::BasicBlock *BB) {
  BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(this, BB));
  BasicBlockNode *Result = BlockNodes.back().get();
  BBMap[BB] = Result;;
  revng_log(CombLogger, "Building " << BB->getName()
                         << " at address: " << BBMap[BB] << "\n");
  return Result;
}

BasicBlockNode *RegionCFG::cloneNode(const BasicBlockNode &OriginalNode) {
  BlockNodes.emplace_back(std::make_unique<BasicBlockNode>(OriginalNode, this));
  BasicBlockNode *New = BlockNodes.back().get();
  New->setName(std::string(OriginalNode.getName()) + " cloned");
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

void
RegionCFG::insertBulkNodes(std::set<BasicBlockNode *> &Nodes,
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

void
RegionCFG::connectBreakNode(std::set<EdgeDescriptor> &Outgoing,
                            BasicBlockNode *Break,
                            const BBNodeMap &SubstitutionMap) {
  for (EdgeDescriptor Edge : Outgoing)
    addEdge(EdgeDescriptor(SubstitutionMap.at(Edge.first), Break));
}

void RegionCFG::connectContinueNode(BasicBlockNode *Continue) {
  std::vector<BasicBlockNode *> ContinueNodes;

  // We need to pre-save the edges to avoid breaking the predecessor iterator
  for (BasicBlockNode *Source : EntryNode->predecessors()) {
    ContinueNodes.push_back(Source);
  }
  for (BasicBlockNode *Source : ContinueNodes) {
    moveEdgeTarget(EdgeDescriptor(Source, EntryNode), Continue);
  }
}

BasicBlockNode &RegionCFG::get(llvm::BasicBlock *BB) {
  auto It = BBMap.find(BB);
  revng_assert(It != BBMap.end());
  return *(It->second);
}

BasicBlockNode &RegionCFG::getRandomNode() {
  int randNum = rand() % (BBMap.size());
  auto randomIt = std::next(std::begin(BBMap), randNum);
  return *(randomIt->second);
}

std::vector<BasicBlockNode *> RegionCFG::orderNodes(std::vector<BasicBlockNode *> &L,
                                              bool DoReverse) {
  llvm::ReversePostOrderTraversal<BasicBlockNode *> RPOT(EntryNode);
  std::vector<BasicBlockNode *> Result;

  if (DoReverse) {
    std::reverse(RPOT.begin(), RPOT.end());
  }

  #if 0
  CombLogger << "New ordering" << "\n";
  for (BasicBlockNode *Node : L) {
    CombLogger << Node->getNameStr() << "\n";
    CombLogger.emit();
  }
  #endif

  for (BasicBlockNode *RPOTBB : RPOT) {
    for (BasicBlockNode *Node : L) {
      if (RPOTBB == Node) {
        Result.push_back(Node);
      }
    }
  }

  revng_assert(L.size() == Result.size());

  return Result;
}

template<typename StreamT>
void RegionCFG::streamNode(StreamT &S, const BasicBlockNode *BB) const {
  unsigned NodeID = BB->getID();
  S << "\"" << NodeID << "\"";
  S << " [" << "label=\"ID: " << NodeID << " Name: " << BB->getName().str() << "\"";
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
      S << "\"" << PredID << "\"" << " -> \"" << SuccID << "\""
        << " [color=green];\n";
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
      if (((*It)->isEmpty())
          and ((*It)->predecessor_size() == 1)
          and ((*It)->successor_size() == 1)) {

        if (CombLogger.isEnabled()) {
          CombLogger << "Purging dummy node " << (*It)->getNameStr() << "\n";
        }

        BasicBlockNode *Predecessor = (*It)->getPredecessorI(0);
        BasicBlockNode *Successor = (*It)->getSuccessorI(0);

        // Connect directly predecessor and successor, and remove the dummy node
        // under analysis
        addEdge(EdgeDescriptor(Predecessor, Successor));
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

std::vector<BasicBlockNode *> RegionCFG::getInterestingNodes(BasicBlockNode *Cond) {

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

  // Apply the comb to a RegionCFG object.
  // TODO: handle all the collapsed regions.
  RegionCFG &Graph = *this;

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
      // else branches are not disjoint
      std::set<BasicBlockNode *> ThenExits = ReachableExits[(*It)->getSuccessorI(0)];
      std::set<BasicBlockNode *> ElseExits = ReachableExits[(*It)->getSuccessorI(1)];
      std::vector<BasicBlockNode *> Intersection;
      std::set_intersection(ThenExits.begin(),
                            ThenExits.end(),
                            ElseExits.begin(),
                            ElseExits.end(),
                            std::back_inserter(Intersection));
      if (Intersection.size() != 0) {
          ConditionalNodes.push_back(*It);
          ConditionalNodesComplete.insert(*It);
      } else {
        CombLogger << "Blacklisted conditional: " << (*It)->getNameStr() << "\n";
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
      //NotDominatedCandidates = getInterestingNodes(Conditional);

      if (CombLogger.isEnabled()) {
        CombLogger << "Analyzing candidate nodes\n ";
      }
      BasicBlockNode *Candidate = NotDominatedCandidates.back();
      NotDominatedCandidates.pop_back();
      if (CombLogger.isEnabled()) {
        CombLogger << "Analyzing candidate " << Candidate->getNameStr()
                   << "\n";
      }

      // Decide wether to insert a dummy or to duplicate.
      if (Candidate->predecessor_size() > 2) {

        // Insert a dummy node.
        if (CombLogger.isEnabled()) {
          CombLogger << "Inserting a dummy node for ";
          CombLogger << Candidate->getNameStr() << "\n";
        }

        typedef enum {Left, Right} Side;

        std::vector<Side> Sides{Left, Right};
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
        assert(Duplicated != nullptr);

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
            moveEdgeTarget(EdgeDescriptor(Predecessor, Candidate),
                           Duplicated);

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

  //if (CombLogger.isEnabled()) {
    CombLogger << "Graph after combing is:\n";
    Graph.dumpDotOnFile("inflates",
                        FunctionName,
                        "Region-" + RegionName + "-after-combing");
  //}
}

void RegionCFG::generateAst() {

  RegionCFG &Graph = *this;

  // Apply combing to the current RegionCFG.
  if (ToInflate) {
    CombLogger << "Inflating region " + RegionName + "\n";
    Graph.inflate();
    ToInflate = false;
  }

  // TODO: factorize out the AST generation phase.
  llvm::DominatorTreeBase<BasicBlockNode, false> ASTDT;
  ASTDT.recalculate(Graph);
#if 0
  llvm::raw_os_ostream Stream(dbg);
#endif
  ASTDT.updateDFSNumbers();
#if 0
  ASTDT.print(Stream);
  Stream.flush();
  ASTDT.print(CombLogger);
#endif

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
    std::vector<llvm::DomTreeNodeBase<BasicBlockNode> *> Children =
      ASTDT[Node]->getChildren();

    std::vector<ASTNode *> ASTChildren;
    for (llvm::DomTreeNodeBase<BasicBlockNode> *TreeNode : Children) {
      BasicBlockNode *BlockNode = TreeNode->getBlock();
      ASTNode *ASTPointer = AST.findASTNode(BlockNode);
      ASTChildren.push_back(ASTPointer);
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
        BodyGraph->generateAst();
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
        BodyGraph->generateAst();
        ASTNode *Body = BodyGraph->getAST().getRoot();
        std::unique_ptr<ASTNode> ASTObject(new ScsNode(Node, Body));
        AST.addASTNode(Node, std::move(ASTObject));
      }
    } else {
      revng_assert(Children.size() < 4);
      std::unique_ptr<ASTNode> ASTObject;
      if (Children.size() == 3) {
        ASTObject.reset(new IfNode(Node, ASTChildren[0], ASTChildren[2],
                                   ASTChildren[1]));
      } else if (Children.size() == 2) {
        ASTObject.reset(new IfNode(Node, ASTChildren[0], ASTChildren[1],
                                   nullptr));
      } else if (Children.size() == 1) {
        revng_assert(not Node->isBreak() and not Node->isContinue());
        ASTObject.reset(new CodeNode(Node, ASTChildren[0]));
      } else if (Children.size() == 0) {
        if (Node->isBreak())
          ASTObject.reset(new BreakNode());
        else if (Node->isContinue())
          ASTObject.reset(new ContinueNode());
        else if (Node->isEmpty() or Node->isCode())
          ASTObject.reset(new CodeNode(Node, nullptr));
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

  // Flip IFs with empty then branches.
  CombLogger << "Performing IFs with empty then branches flipping\n";
  flipEmptyThen(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-if-flip");

  #if 0
  // Simplify short-circuit nodes.
  CombLogger << "Performing short-circuit simplification\n";
  simplifyShortCircuit(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-short-circuit");
  #endif

  // Remove danling nodes (possibly created by the de-optimization pass, after
  // disconnecting the first CFG node corresponding to the simplified AST node),
  // and superfluos dummy nodes
  removeNotReachables();
  purgeDummies();

  // TODO: Remove or change this
  dumpDotOnFile("deoptimizes", FunctionName, "Deoptimized-cfg-" + RegionName);

  #if 0
  // Simplify trivial short-circuit nodes.
  CombLogger << "Performing trivial short-circuit simplification\n";
  simplifyTrivialShortCircuit(RootNode);
  AST.dumpOnFile("ast", FunctionName, "After-trivial-short-circuit");
  #endif
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
