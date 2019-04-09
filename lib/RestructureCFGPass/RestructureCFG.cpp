/// \file Restructure.cpp
/// \brief FunctionPass that applies the comb to the RegionCFG of a function

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <sstream>
#include <stdlib.h>

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"

// revng includes
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// Local includes
#include "Flattening.h"
#include "MetaRegion.h"

using namespace llvm;

using std::make_pair;
using std::pair;
using std::string;
using std::to_string;

// TODO: Move the initialization of the logger here from "Utils.h"
// Debug logger.
Logger<> CombLogger("restructure");

// EdgeDescriptor is a handy way to create and manipulate edges on the
// RegionCFG.
using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;

static std::set<EdgeDescriptor> getBackedges(RegionCFG &Graph) {

  // Some helper data structures.
  int Time = 0;
  std::map<BasicBlockNode *, int> StartTime;
  std::map<BasicBlockNode *, int> FinishTime;
  std::vector<std::pair<BasicBlockNode *, size_t>> Stack;

  // Set of backedges.
  std::set<EdgeDescriptor> Backedges;

  // Push the entry node in the exploration stack.
  BasicBlockNode &EntryNode = Graph.getEntryNode();
  Stack.push_back(make_pair(&EntryNode, 0));

  // Go through the exploration stack.
  while (!Stack.empty()) {
    auto StackElem = Stack.back();
    Stack.pop_back();
    BasicBlockNode *Vertex = StackElem.first;
    Time++;

    // Check if we are inspecting a vertex for the first time, and in case mark
    // the start time of the visit.
    if (StartTime.count(Vertex) == 0) {
      StartTime[Vertex] = Time;
    }

    // Successor exploraition
    size_t Index = StackElem.second;

    // If we are still successors to explore.
    if (Index < StackElem.first->successor_size()) {
      BasicBlockNode *Successor = Vertex->getSuccessorI(Index);
      Index++;
      Stack.push_back(make_pair(Vertex, Index));

      // We are in presence of a backedge.
      if (StartTime.count(Successor) != 0
          and FinishTime.count(Successor) == 0) {
        Backedges.insert(make_pair(Vertex, Successor));
      }

      // Enqueue the successor for the visit.
      if (StartTime.count(Successor) == 0) {
        Stack.push_back(make_pair(Successor, 0));
      }
    } else {

      // Mark the finish of the visit of a vertex.
      FinishTime[Vertex] = Time;
    }
  }

  return Backedges;
}

static bool mergeSCSStep(std::vector<MetaRegion> &MetaRegions) {
  for (auto RegionIt1 = MetaRegions.begin(); RegionIt1 != MetaRegions.end();
       RegionIt1++) {
    for (auto RegionIt2 = std::next(RegionIt1); RegionIt2 != MetaRegions.end();
         RegionIt2++) {
      bool Intersects = (*RegionIt1).intersectsWith(*RegionIt2);
      bool IsIncluded = (*RegionIt1).isSubSet(*RegionIt2);
      bool IsIncludedReverse = (*RegionIt2).isSubSet(*RegionIt1);
      bool AreEquivalent = (*RegionIt1).nodesEquality(*RegionIt2);
      if (Intersects
          and (((!IsIncluded) and (!IsIncludedReverse)) or AreEquivalent)) {
        (*RegionIt1).mergeWith(*RegionIt2);
        MetaRegions.erase(RegionIt2);
        return true;
      }
    }
  }

  return false;
}

static void simplifySCS(std::vector<MetaRegion> &MetaRegions) {
  bool Changes = true;
  while (Changes) {
    Changes = mergeSCSStep(MetaRegions);
  }
}

static void sortMetaRegions(std::vector<MetaRegion> &MetaRegions) {
  std::sort(MetaRegions.begin(),
            MetaRegions.end(),
            [](MetaRegion &First, MetaRegion &Second) {
              return First.getNodes().size() < Second.getNodes().size();
            });
}

static void computeParents(std::vector<MetaRegion> &MetaRegions,
                           MetaRegion *RootMetaRegion) {
  for (MetaRegion &MetaRegion1 : MetaRegions) {
    bool ParentFound = false;
    for (MetaRegion &MetaRegion2 : MetaRegions) {
      if (&MetaRegion1 != &MetaRegion2) {
        if (MetaRegion1.isSubSet(MetaRegion2)) {

          if (CombLogger.isEnabled()) {
            CombLogger << "For metaregion: " << &MetaRegion1 << "\n";
            CombLogger << "parent found\n";
            CombLogger << &MetaRegion2 << "\n";
          }

          MetaRegion1.setParent(&MetaRegion2);
          ParentFound = true;
          break;
        }
      }
    }

    if (!ParentFound) {

      if (CombLogger.isEnabled()) {
        CombLogger << "For metaregion: " << &MetaRegion1 << "\n";
        CombLogger << "no parent found\n";
      }

      MetaRegion1.setParent(RootMetaRegion);
    }
  }
}

static std::vector<MetaRegion *> applyPartialOrder(std::vector<MetaRegion> &V) {
  std::vector<MetaRegion *> OrderedVector;
  std::set<MetaRegion *> Processed;

  while (V.size() != Processed.size()) {
    for (auto RegionIt1 = V.begin(); RegionIt1 != V.end(); RegionIt1++) {
      if (Processed.count(&*RegionIt1) == 0) {
        bool FoundParent = false;
        for (auto RegionIt2 = V.begin(); RegionIt2 != V.end(); RegionIt2++) {
          if ((RegionIt1 != RegionIt2) and Processed.count(&*RegionIt2) == 0) {
            if ((*RegionIt1).getParent() == &*RegionIt2) {
              FoundParent = true;
              break;
            }
          }
        }

        if (FoundParent == false) {
          OrderedVector.push_back(&*RegionIt1);
          Processed.insert(&*RegionIt1);
          break;
        }
      }
    }
  }

  std::reverse(OrderedVector.begin(), OrderedVector.end());
  return OrderedVector;
}

static bool alreadyInMetaregion(std::vector<MetaRegion> &V, BasicBlockNode *N) {

  // Scan all the metaregions and check if a node is already contained in one of
  // them
  for (MetaRegion &Region : V) {
    if (Region.containsNode(N)) {
      return true;
    }
  }

  return false;
}

static std::vector<MetaRegion>
createMetaRegions(const std::set<EdgeDescriptor> &Backedges) {
  std::map<BasicBlockNode *, std::set<BasicBlockNode *>> AdditionalSCSNodes;
  std::vector<std::pair<BasicBlockNode *, std::set<BasicBlockNode *>>> Regions;
  for (auto &Backedge : Backedges) {
    auto SCSNodes = findReachableNodes(*Backedge.second, *Backedge.first);
    AdditionalSCSNodes[Backedge.second].insert(SCSNodes.begin(),
                                               SCSNodes.end());

    if (CombLogger.isEnabled()) {
      CombLogger << "SCS identified by: ";
      CombLogger << Backedge.first->getNameStr() << " -> "
                 << Backedge.second->getNameStr() << "\n";
      CombLogger << "Is composed of nodes:\n";
      for (auto Node : SCSNodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }

    Regions.push_back(std::make_pair(Backedge.second, SCSNodes));
  }

  // Include in the regions found before other possible sub-regions, if an edge
  // which is the target of a backedge is included in an outer region.
  for (auto &Region : Regions) {
    BasicBlockNode *Head = Region.first;
    std::set<BasicBlockNode *> &Nodes = Region.second;
    for (BasicBlockNode *Node : Nodes) {
      if ((Node != Head) and (AdditionalSCSNodes.count(Node) != 0)) {
        CombLogger << "Adding additional nodes for region with head: ";
        CombLogger << Head->getNameStr();
        CombLogger << " and relative to node: ";
        CombLogger << Node->getNameStr() << "\n";
        Nodes.insert(AdditionalSCSNodes[Node].begin(),
                     AdditionalSCSNodes[Node].end());
      }
    }
  }

  std::vector<MetaRegion> MetaRegions;
  int SCSIndex = 1;
  for (size_t I = 0; I < Regions.size(); ++I) {
    auto &SCS = Regions[I].second;
    MetaRegions.push_back(MetaRegion(SCSIndex, SCS, true));
    SCSIndex++;
  }
  return MetaRegions;
}

char RestructureCFG::ID = 0;
static RegisterPass<RestructureCFG> X("restructure-cfg",
                                      "Apply RegionCFG restructuring "
                                      "transformation",
                                      true,
                                      true);

bool RestructureCFG::runOnFunction(Function &F) {

  // Analyze only isolated functions.
  if (!F.getName().startswith("bb.")
      or F.getName().startswith("bb.quotearg_buffer_restyled")
      or F.getName().startswith("bb._getopt_internal_r")
      or F.getName().startswith("bb.printf_parse")
      or F.getName().startswith("bb.vasnprintf")) {
    return false;
  }

  // Clear graph object from the previous pass.
  RootCFG = RegionCFG();

  // Set names of the CFG region
  RootCFG.setFunctionName(F.getName());
  RootCFG.setRegionName("root");

  // Random seed initialization
  srand(time(NULL));

  // Initialize the RegionCFG object
  RootCFG.initialize(F);

  // Dump the function name.
  if (CombLogger.isEnabled()) {
    CombLogger << "Analyzing function: " << F.getName() << "\n";
  }

  // Dump the object in .dot format if debug mode is activated.
  if (CombLogger.isEnabled()) {
    RootCFG.dumpDotOnFile("dots", F.getName(), "begin");
  }

  // Identify SCS regions.
  if (CombLogger.isEnabled()) {
    BasicBlockNode &FirstRandom = RootCFG.getRandomNode();
    BasicBlockNode &SecondRandom = RootCFG.getRandomNode();
    CombLogger << "Source: ";
    CombLogger << FirstRandom.getNameStr() << "\n";
    CombLogger << "Target: ";
    CombLogger << SecondRandom.getNameStr() << "\n";
    CombLogger << "Nodes Reachable:\n";
    std::set<BasicBlockNode *> Reachables = findReachableNodes(FirstRandom,
                                                               SecondRandom);
    for (BasicBlockNode *Element : Reachables) {
      CombLogger << Element->getNameStr() << "\n";
    }
  }

  std::set<EdgeDescriptor> Backedges = getBackedges(RootCFG);
  CombLogger << "Backedges in the graph:\n";
  for (auto &Backedge : Backedges) {
    CombLogger << Backedge.first->getNameStr() << " -> "
               << Backedge.second->getNameStr() << "\n";
  }

  // Create meta regions
  std::vector<MetaRegion> MetaRegions = createMetaRegions(Backedges);

  // Simplify SCS in a fixed-point fashion.
  simplifySCS(MetaRegions);

  // Print SCS after simplification.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after simplification:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      auto &Nodes = Meta.getNodes();
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }
  }

  // Sort the Metaregions in increasing number of composing nodes order.
  sortMetaRegions(MetaRegions);

  // Print SCS after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after ordering:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }
  }

  // Compute parent relations for the identified SCSs.
  std::set<BasicBlockNode *> Empty;
  MetaRegion RootMetaRegion(0, Empty);
  computeParents(MetaRegions, &RootMetaRegion);

  // Print metaregions after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions parent relationship:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      auto &Nodes = Meta.getNodes();
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
      CombLogger << "Has parent: " << Meta.getParent() << "\n";
    }
  }

  // Find an ordering for the metaregions that satisfies the inclusion
  // relationship. We create a new "shadow" vector containing only pointers to
  // the "real" metaregions.
  std::vector<MetaRegion *> OrderedMetaRegions = applyPartialOrder(MetaRegions);

  // Print metaregions after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after ordering:\n";
    for (auto *Meta : OrderedMetaRegions) {
      CombLogger << "\n";
      CombLogger << Meta << "\n";
      CombLogger << "With index " << Meta->getIndex() << "\n";
      CombLogger << "With size " << Meta->nodes_size() << "\n";
      auto &Nodes = Meta->getNodes();
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
      CombLogger << "Has parent: " << Meta->getParent() << "\n";
      CombLogger << "Is SCS: " << Meta->isSCS() << "\n";
    }
  }

  ReversePostOrderTraversal<BasicBlockNode *> RPOT(&RootCFG.getEntryNode());
  if (CombLogger.isEnabled()) {
    CombLogger << "Reverse post order is:\n";
    for (BasicBlockNode *BN : RPOT) {
      CombLogger << BN->getNameStr() << "\n";
    }
    CombLogger << "Reverse post order end\n";
  }

  CombLogger << "Debugged function"
             << "\n";
  CombLogger << F.getName().equals("bb._start_c") << "\n";

  DominatorTreeBase<BasicBlockNode, false> DT;
  DT.recalculate(RootCFG);

  DominatorTreeBase<BasicBlockNode, true> PDT;
  PDT.recalculate(RootCFG);

  // Some debug information on dominator and postdominator tree.
  if (CombLogger.isEnabled()) {
    CombLogger << DT.isPostDominator() << "\n";
    CombLogger << "The root node of the dominator tree is:\n";
    CombLogger << DT.getRoot()->getNameStr() << "\n";
    CombLogger << "Between these two nodes:\n";
    BasicBlockNode *Random = &RootCFG.getRandomNode();
    BasicBlockNode *Random2 = &RootCFG.getRandomNode();
    CombLogger << Random->getNameStr() << "\n";
    CombLogger << Random2->getNameStr() << "\n";
    CombLogger << "Dominance:\n";
    CombLogger << DT.dominates(Random, Random2) << "\n";
    CombLogger << "PostDominance:\n";
    CombLogger << PDT.dominates(Random, Random2) << "\n";
    CombLogger << PDT.isPostDominator() << "\n";
  }

  // Reserve enough space for all the OrderedMetaRegions.
  // The following algorithms stores pointers to the elements of this vector, so
  // we need to make sure that no reallocation happens.
  std::vector<RegionCFG> Regions(OrderedMetaRegions.size());

  for (MetaRegion *Meta : OrderedMetaRegions) {
    if (CombLogger.isEnabled()) {
      CombLogger << "\nAnalyzing region: " << Meta->getIndex() << "\n";
    }

    // Refresh backedges, since some of them may have been modified during
    // the transformations
    Backedges = getBackedges(RootCFG);

    if (CombLogger.isEnabled()) {

      auto &Nodes = Meta->getNodes();
      CombLogger << "Which is composed of nodes:\n";
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }

      CombLogger << "Dumping main graph snapshot before restructuring\n";
      RootCFG.dumpDotOnFile("dots",
                            F.getName(),
                            "Out-pre-" + std::to_string(Meta->getIndex()));
    }

    std::map<BasicBlockNode *, int> IncomingDegree;
    for (BasicBlockNode *Node : Meta->nodes()) {
      int IncomingCounter = 0;
      for (BasicBlockNode *Predecessor : Node->predecessors()) {
        EdgeDescriptor Edge = make_pair(Predecessor, Node);
        if ((Meta->containsNode(Predecessor)) and (Backedges.count(Edge))) {
          IncomingCounter++;
        }
      }
      IncomingDegree[Node] = IncomingCounter;
    }

    // Print information about incoming edge degrees.
    if (CombLogger.isEnabled()) {
      CombLogger << "Incoming degree:\n";
      for (auto &it : IncomingDegree) {
        CombLogger << it.first->getNameStr() << " " << it.second << "\n";
      }
    }

    auto MaxDegreeIt = max_element(IncomingDegree.begin(),
                                   IncomingDegree.end(),
                                   [](const pair<BasicBlockNode *, int> &p1,
                                      const pair<BasicBlockNode *, int> &p2) {
                                     return p1.second < p2.second;
                                   });
    int MaxDegree = (*MaxDegreeIt).second;

    if (CombLogger.isEnabled()) {
      CombLogger << "Maximum incoming degree found: ";
      CombLogger << MaxDegree << "\n";
    }

    std::set<BasicBlockNode *> MaximuxEdgesNodes;
    copy_if(Meta->begin(),
            Meta->end(),
            std::inserter(MaximuxEdgesNodes, MaximuxEdgesNodes.begin()),
            [&IncomingDegree, &MaxDegree](BasicBlockNode *Node) {
              return IncomingDegree[Node] == MaxDegree;
            });

    revng_assert(MaxDegree > 0);

    BasicBlockNode *FirstCandidate;
    if (MaximuxEdgesNodes.size() > 1) {
      for (BasicBlockNode *BN : RPOT) {
        if (MaximuxEdgesNodes.count(BN) != 0) {
          FirstCandidate = BN;
          break;
        }
      }
    } else {
      FirstCandidate = *MaximuxEdgesNodes.begin();
    }

    revng_assert(FirstCandidate != nullptr);

    // Print out the name of the node that has been selected as head of the
    // region
    if (CombLogger.isEnabled()) {
      CombLogger << "Elected head is: " << FirstCandidate->getNameStr() << "\n";
    }

    // Identify all the abnormal retreating edges in a SCS.
    std::set<EdgeDescriptor> Retreatings;
    std::set<BasicBlockNode *> RetreatingTargets;
    for (EdgeDescriptor Backedge : Backedges) {
      if (Meta->containsNode(Backedge.first)) {
        Retreatings.insert(Backedge);
        RetreatingTargets.insert(Backedge.second);
      }
    }
    if (CombLogger.isEnabled()) {
      CombLogger << "Retreatings found:\n";
      for (EdgeDescriptor Retreating : Retreatings) {
        CombLogger << Retreating.first->getNameStr() << " -> ";
        CombLogger << Retreating.second->getNameStr() << "\n";
      }
    }

    bool NewHeadNeeded = false;
    for (BasicBlockNode *Node : RetreatingTargets) {
      if (Node != FirstCandidate) {
        NewHeadNeeded = true;
      }
    }
    if (CombLogger.isEnabled()) {
      CombLogger << "New head needed: " << NewHeadNeeded << "\n";
    }

    BasicBlockNode *Head;
    if (NewHeadNeeded) {
      revng_assert(RetreatingTargets.size() > 1);
      std::map<BasicBlockNode *, int> RetreatingIdxMap;

      BasicBlockNode *const False = *RetreatingTargets.begin();
      RetreatingIdxMap[False] = 0;

      BasicBlockNode *const True = *std::next(RetreatingTargets.begin());
      RetreatingIdxMap[True] = 1;

      unsigned Idx = 1;
      Head = RootCFG.addDispatcher(Idx, True, False);
      Meta->insertNode(Head);

      Idx = 2;
      using TargetIterator = std::set<BasicBlockNode *>::iterator;
      TargetIterator TgtIt = std::next(std::next(RetreatingTargets.begin()));
      TargetIterator TgtEnd = RetreatingTargets.end();
      for (; TgtIt != TgtEnd; ++TgtIt) {
        BasicBlockNode *New = RootCFG.addDispatcher(Idx, *TgtIt, Head);
        Meta->insertNode(New);
        RetreatingIdxMap[*TgtIt] = Idx;
        Idx++;
        Head = New;
      }
      revng_assert(Idx == RetreatingTargets.size());

      for (EdgeDescriptor R : Retreatings) {
        Idx = RetreatingIdxMap[R.second];
        auto *SetNode = RootCFG.addSetStateNode(Idx, R.second->getName());
        Meta->insertNode(SetNode);
        moveEdgeTarget(EdgeDescriptor(R.first, R.second), SetNode);
        addEdge(EdgeDescriptor(SetNode, Head));
      }

      // Move the incoming edge from the old head to new one.
      for (BasicBlockNode *Predecessor : FirstCandidate->predecessors()) {
        if (!Meta->containsNode(Predecessor)) {
          moveEdgeTarget(EdgeDescriptor(Predecessor, FirstCandidate), Head);
        }
      }

    } else {
      Head = FirstCandidate;
    }

    revng_assert(Head != nullptr);
    if (CombLogger.isEnabled()) {
      CombLogger << "New head name is: " << Head->getNameStr() << "\n";
    }

    // Successor refinement step.
    std::set<BasicBlockNode *> Successors = Meta->getSuccessors();

    if (CombLogger.isEnabled()) {
      CombLogger << "Region successors are:\n";
      for (BasicBlockNode *Node : Successors) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }

    bool AnotherIteration = true;
    while (AnotherIteration and Successors.size() > 1) {
      AnotherIteration = false;
      std::set<EdgeDescriptor> OutgoingEdges = Meta->getOutEdges();

      std::vector<BasicBlockNode *> Frontiers;
      std::map<BasicBlockNode *, pair<BasicBlockNode *, BasicBlockNode *>>
        EdgeExtremal;

      for (EdgeDescriptor Edge : OutgoingEdges) {
        BasicBlockNode *Frontier = RootCFG.addArtificialNode("frontier");
        BasicBlockNode *OldSource = Edge.first;
        BasicBlockNode *OldTarget = Edge.second;
        EdgeExtremal[Frontier] = make_pair(OldSource, OldTarget);
        moveEdgeTarget(Edge, Frontier);
        addEdge(EdgeDescriptor(Frontier, OldTarget));
        Meta->insertNode(Frontier);
        Frontiers.push_back(Frontier);
      }

      DT.recalculate(RootCFG);
      for (BasicBlockNode *Frontier : Frontiers) {
        for (BasicBlockNode *Successor : Successors) {
          if ((DT.dominates(Head, Successor))
              and (DT.dominates(Frontier, Successor))
              and !alreadyInMetaregion(MetaRegions, Successor)) {
            Meta->insertNode(Successor);
            AnotherIteration = true;
            if (CombLogger.isEnabled()) {
              CombLogger << "Identified new candidate for successor "
                            "refinement:";
              CombLogger << Successor->getNameStr() << "\n";
            }
          }
        }
      }

      // Remove the frontier nodes since we do not need them anymore.
      for (BasicBlockNode *Frontier : Frontiers) {
        EdgeDescriptor Extremal = EdgeExtremal[Frontier];
        BasicBlockNode *OriginalSource = EdgeExtremal[Frontier].first;
        BasicBlockNode *OriginalTarget = EdgeExtremal[Frontier].second;
        moveEdgeTarget({ OriginalSource, Frontier }, OriginalTarget);
        RootCFG.removeNode(Frontier);
        Meta->removeNode(Frontier);
      }

      Successors = Meta->getSuccessors();
    }

    // First Iteration outlining.
    // Clone all the nodes of the SCS except for the head.
    std::map<BasicBlockNode *, BasicBlockNode *> ClonedMap;
    for (BasicBlockNode *Node : Meta->nodes()) {
      if (Node != Head) {
        BasicBlockNode *Clone = RootCFG.cloneNode(*Node);
        ClonedMap[Node] = Clone;
      }
    }

    // Restore edges between cloned nodes.
    for (BasicBlockNode *Node : Meta->nodes()) {
      if (Node != Head) {

        // Handle outgoing edges from SCS nodes.
        if (Node->isCheck()) {

          if (Meta->containsNode(Node->getTrue()))
            ClonedMap.at(Node)->setTrue(ClonedMap.at(Node->getTrue()));
          else
            ClonedMap.at(Node)->setTrue(Node->getTrue());

          if (Meta->containsNode(Node->getFalse()))
            ClonedMap.at(Node)->setFalse(ClonedMap.at(Node->getFalse()));
          else
            ClonedMap.at(Node)->setFalse(Node->getFalse());

        } else {
          for (BasicBlockNode *Successor : Node->successors()) {
            if (Meta->containsNode(Successor)) {
              // Handle edges pointing inside the SCS.
              if ((Successor == Head) or (Successor == FirstCandidate)) {
                // Retreating edges should point to the new head.
                addEdge(EdgeDescriptor(ClonedMap[Node], Head));
              } else {
                // Other edges should be restored between cloned nodes.
                addEdge(EdgeDescriptor(ClonedMap[Node], ClonedMap[Successor]));
              }
            } else {
              // Edges exiting from the SCS should go to the right target.
              addEdge(EdgeDescriptor(ClonedMap[Node], Successor));
            }
          }
        }

        // We need this temporary vector to avoid invalidating iterators.
        std::vector<BasicBlockNode *> Predecessors;
        for (BasicBlockNode *Predecessor : Node->predecessors()) {
          Predecessors.push_back(Predecessor);
        }
        for (BasicBlockNode *Predecessor : Predecessors) {
          if (!Meta->containsNode(Predecessor)) {
            moveEdgeTarget(EdgeDescriptor(Predecessor, Node), ClonedMap[Node]);
          }
        }
      }
    }

    if (NewHeadNeeded) {
      revng_assert(Head->isCheck());
      std::set<BasicBlockNode *> SetCandidates;
      for (BasicBlockNode *Pred : Head->predecessors()) {
        if (not Pred->isSet()) {
          SetCandidates.insert(Pred);
        }
      }
      unsigned Value = RetreatingTargets.size() - 1;
      for (BasicBlockNode *Pred : SetCandidates) {
        BasicBlockNode *Set = RootCFG.addSetStateNode(Value, Head->getName());
        addEdge(EdgeDescriptor(Pred, Set));
        addEdge(EdgeDescriptor(Set, Head));
        removeEdge(EdgeDescriptor(Pred, Head));
      }
    }

    // Exit dispatcher creation.
    // TODO: Factorize this out together with the head dispatcher creation.
    bool NewExitNeeded = false;
    BasicBlockNode *Exit;
    std::vector<BasicBlockNode *> ExitDispatcherNodes;
    if (Successors.size() > 1) {
      NewExitNeeded = true;
    }
    if (CombLogger.isEnabled()) {
      CombLogger << "New exit needed: " << NewExitNeeded << "\n";
    }

    if (NewExitNeeded) {
      revng_assert(Successors.size() > 1);
      std::map<BasicBlockNode *, int> SuccessorsIdxMap;

      BasicBlockNode *const False = *Successors.begin();
      SuccessorsIdxMap[False] = 0;

      BasicBlockNode *const True = *std::next(Successors.begin());
      SuccessorsIdxMap[True] = 1;

      unsigned Idx = 1;
      Exit = RootCFG.addDispatcher(Idx, True, False);
      ExitDispatcherNodes.push_back(Exit);

      Idx = 2;
      using SuccessorIterator = std::set<BasicBlockNode *>::iterator;
      SuccessorIterator SuccIt = std::next(std::next(Successors.begin()));
      SuccessorIterator SuccEnd = Successors.end();
      for (; SuccIt != SuccEnd; ++SuccIt) {
        BasicBlockNode *New = RootCFG.addDispatcher(Idx, *SuccIt, Exit);
        ExitDispatcherNodes.push_back(New);
        SuccessorsIdxMap[*SuccIt] = Idx;
        Idx++;
        Exit = New;
      }
      revng_assert(Idx == Successors.size());

      std::set<EdgeDescriptor> OutEdges = Meta->getOutEdges();
      for (EdgeDescriptor Edge : OutEdges) {
        Idx = SuccessorsIdxMap.at(Edge.second);
        auto *IdxSetNode = RootCFG.addSetStateNode(Idx, Edge.second->getName());
        Meta->insertNode(IdxSetNode);
        moveEdgeTarget(EdgeDescriptor(Edge.first, Edge.second), IdxSetNode);
        addEdge(EdgeDescriptor(IdxSetNode, Edge.second));
      }
      if (CombLogger.isEnabled()) {
        CombLogger << "New exit name is: " << Exit->getNameStr() << "\n";
      }
    }

    // Collapse Region.
    // Create a new RegionCFG object for representing the collapsed region and
    // populate it with the internal nodes.
    Regions.push_back(RegionCFG());
    RegionCFG &CollapsedGraph = Regions.back();
    RegionCFG::BBNodeMap SubstitutionMap{};
    CollapsedGraph.setFunctionName(F.getName());
    CollapsedGraph.setRegionName(std::to_string(Meta->getIndex()));
    revng_assert(Head != nullptr);
    CollapsedGraph.insertBulkNodes(Meta->getNodes(), Head, SubstitutionMap);

    // Create the break and continue node.
    BasicBlockNode *Continue = CollapsedGraph.addContinue();
    BasicBlockNode *Break = CollapsedGraph.addBreak();

    // Connect the break and continue nodes with the necessary edges.
    CollapsedGraph.connectContinueNode(Continue);
    std::set<EdgeDescriptor> OutgoingEdges = Meta->getOutEdges();
    CollapsedGraph.connectBreakNode(OutgoingEdges, Break, SubstitutionMap);

    // Create the collapsed node in the outer region.
    BasicBlockNode *Collapsed = RootCFG.createCollapsedNode(&CollapsedGraph);

    // Connect the old incoming edges to the collapsed node.
    std::set<EdgeDescriptor> IncomingEdges = Meta->getInEdges();
    for (EdgeDescriptor Edge : IncomingEdges)
      moveEdgeTarget(Edge, Collapsed);

    // Connect the outgoing edges to the collapsed node.
    if (NewExitNeeded) {
      revng_assert(Exit != nullptr);
      addEdge(EdgeDescriptor(Collapsed, Exit));
    } else {

      // Double check that we have at most a single successor
      revng_assert(Successors.size() <= 1);
      if (Successors.size() == 1) {

        // Connect the collapsed node to the unique successor
        BasicBlockNode *Successor = *Successors.begin();
        addEdge(EdgeDescriptor(Collapsed, Successor));
      }
    }

    // Remove collapsed nodes from the outer region.
    for (BasicBlockNode *Node : Meta->nodes()) {
      if (CombLogger.isEnabled()) {
        CombLogger << "Removing from main graph node :" << Node->getNameStr()
                   << "\n";
      }
      RootCFG.removeNode(Node);
    }

    // Substitute in the other SCSs the nodes of the current SCS with the
    // collapsed node and the exit dispatcher structure.
    for (MetaRegion *OtherMeta : OrderedMetaRegions) {
      if (OtherMeta != Meta) {
        OtherMeta->updateNodes(Meta->getNodes(),
                               Collapsed,
                               ExitDispatcherNodes);
      }
    }

    // Replace the pointers inside SCS.
    Meta->replaceNodes(CollapsedGraph.getNodes());

    // Remove useless nodes inside the SCS (like dandling break/continue)
    CollapsedGraph.removeNotReachables();

    // Serialize the newly collapsed SCS region.
    if (CombLogger.isEnabled()) {
      CombLogger << "Dumping CFG of metaregion " << Meta->getIndex() << "\n";
      CollapsedGraph.dumpDotOnFile("dots",
                                   F.getName(),
                                   "In-" + std::to_string(Meta->getIndex()));
      CombLogger << "Dumping main graph snapshot post restructuring\n";
      RootCFG.dumpDotOnFile("dots",
                            F.getName(),
                            "Out-post-" + std::to_string(Meta->getIndex()));
    }

    // Remove not reachables nodes from the graph at each iteration.
    RootCFG.removeNotReachables();
  }

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph before final purge\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "Final-before-purge");
  }

  // Remove not reachables nodes from the main final graph.
  RootCFG.removeNotReachables();

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph after final purge\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "Final-after-purge");
  }

  // Print metaregions after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after collapse:\n";
    for (auto *Meta : OrderedMetaRegions) {
      CombLogger << "\n";
      CombLogger << Meta << "\n";
      CombLogger << "With index " << Meta->getIndex() << "\n";
      CombLogger << "With size " << Meta->nodes_size() << "\n";
      auto &Nodes = Meta->getNodes();
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
      CombLogger << "Has parent: " << Meta->getParent() << "\n";
      CombLogger << "Is SCS: " << Meta->isSCS() << "\n";
    }
  }

  // Invoke the AST generation for the root region.
  CombLogger.emit();
  RootCFG.generateAst();

  // Serialize final AST on file
  RootCFG.getAST().dumpOnFile("ast", F.getName(), "Final");

  // Sync Logger.
  CombLogger.emit();

  // Early exit if the AST generation produced a version of the AST which is
  // identical to the cached version.
  // In that case there's no need to flatten the RegionCFG.
  // TODO: figure out how to decide when we're done
  if (Done)
    return false;

  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph after Flattening\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "final-before-flattening");
  }

  flattenRegionCFGTree(RootCFG);

  // Serialize final AST after flattening on file
  RootCFG.getAST().dumpOnFile("ast", F.getName(), "Final-after-flattening");

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph after Flattening\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "final-after-flattening");
  }

  return false;
}
