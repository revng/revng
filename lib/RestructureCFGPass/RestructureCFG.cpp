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

// revng-c includes
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

// Local libraries includes
#include "revng-c/RestructureCFGPass/FlatteningBB.h"
#include "revng-c/RestructureCFGPass/MetaRegionBB.h"
#include "revng-c/RestructureCFGPass/RegionCFGTreeBB.h"
#include "revng-c/RestructureCFGPass/RestructureCFG.h"
#include "revng-c/RestructureCFGPass/Utils.h"

using namespace llvm;
using namespace llvm::cl;

using std::make_pair;
using std::pair;
using std::string;
using std::to_string;

// TODO: Move the initialization of the logger here from "Utils.h"
// Debug logger.
Logger<> CombLogger("restructure");

// EdgeDescriptor is a handy way to create and manipulate edges on the
// RegionCFG.
using BasicBlockNodeBB = BasicBlockNode<BasicBlock *>;
using EdgeDescriptor = std::pair<BasicBlockNodeBB *, BasicBlockNodeBB *>;

// Explicit instantation of template classes `Metaregion` and `RegionCFG`.
template class MetaRegion<BasicBlock *>;
template class RegionCFG<BasicBlock *>;
using MetaRegionBB = MetaRegion<BasicBlock *>;
using MetaRegionBBVect = std::vector<MetaRegionBB>;
using MetaRegionBBPtrVect = std::vector<MetaRegionBB *>;
using BackedgeMetaRegionMap = std::map<EdgeDescriptor, MetaRegionBB *>;

static std::set<EdgeDescriptor> getBackedges(RegionCFG<BasicBlock *> &Graph) {

  // Some helper data structures.
  int Time = 0;
  std::map<BasicBlockNodeBB *, int> StartTime;
  std::map<BasicBlockNodeBB *, int> FinishTime;
  std::vector<std::pair<BasicBlockNodeBB *, size_t>> Stack;

  // Set of backedges.
  std::set<EdgeDescriptor> Backedges;

  // Push the entry node in the exploration stack.
  BasicBlockNodeBB &EntryNode = Graph.getEntryNode();
  Stack.push_back(make_pair(&EntryNode, 0));

  // Go through the exploration stack.
  while (!Stack.empty()) {
    auto StackElem = Stack.back();
    Stack.pop_back();
    BasicBlockNodeBB *Vertex = StackElem.first;
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
      BasicBlockNodeBB *Successor = Vertex->getSuccessorI(Index);
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

static bool mergeSCSStep(MetaRegionBBVect &MetaRegions) {
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

static void simplifySCS(MetaRegionBBVect &MetaRegions) {
  bool Changes = true;
  while (Changes) {
    Changes = mergeSCSStep(MetaRegions);
  }
}

static bool
mergeSCSAbnormalRetreating(MetaRegionBBVect &MetaRegions,
                           const std::set<EdgeDescriptor> &Backedges,
                           BackedgeMetaRegionMap &BackedgeMetaRegionMap,
                           std::set<MetaRegionBB *> &BlacklistedMetaregions) {
  for (auto RegionIt = MetaRegions.begin(); RegionIt != MetaRegions.end();
       RegionIt++) {
    MetaRegionBB &Region = *RegionIt;

    // Do not re-analyze blacklisted metaregions.
    if (BlacklistedMetaregions.count(&Region) == 0) {

      // Iterate over all the backedges present in the graph, if the current
      // region contains the source of a backedge, it should contain also the
      // the target of that backedge. If not, merge the two SCSs.
      for (EdgeDescriptor Backedge : Backedges) {
        bool FirstIn = Region.containsNode(Backedge.first);
        bool SecondIn = Region.containsNode(Backedge.second);
        bool AbnormalIncoming = FirstIn and not SecondIn;
        bool AbnormalOutgoing = not FirstIn and SecondIn;
        if (AbnormalIncoming or AbnormalOutgoing) {

          // Retrieve the Metaregion identified by the backedge with goes
          // goes outside the scope of the current Metaregion.
          MetaRegionBB *OtherRegion = BackedgeMetaRegionMap.at(Backedge);
          Region.mergeWith(*OtherRegion);

          // Blacklist the region which we have merged.
          BackedgeMetaRegionMap[Backedge] = &Region;
          BlacklistedMetaregions.insert(OtherRegion);
          return true;

          // Abort if we didn't find the metaregion to remove.
          revng_abort("Not found the region to merge with.");
        }
      }
    }
  }

  return false;
}

static void
simplifySCSAbnormalRetreating(MetaRegionBBVect &MetaRegions,
                              const std::set<EdgeDescriptor> &Backedges,
                              BackedgeMetaRegionMap &BackedgeMetaRegionMap) {
  std::set<MetaRegionBB *> BlacklistedMetaregions;
  bool Changes = true;
  while (Changes) {
    Changes = mergeSCSAbnormalRetreating(MetaRegions,
                                         Backedges,
                                         BackedgeMetaRegionMap,
                                         BlacklistedMetaregions);
  }

  // Remove all the metaregion that have been merged with others, using the
  // erase/remove idiom.
  MetaRegions.erase(remove_if(MetaRegions.begin(),
                              MetaRegions.end(),
                              [&BlacklistedMetaregions](MetaRegionBB &M) {
                                return BlacklistedMetaregions.count(&M) == 1;
                              }),
                    MetaRegions.end());
}

static void sortMetaRegions(MetaRegionBBVect &MetaRegions) {
  std::sort(MetaRegions.begin(),
            MetaRegions.end(),
            [](MetaRegionBB &First, MetaRegionBB &Second) {
              return First.getNodes().size() < Second.getNodes().size();
            });
}

static bool
checkMetaregionConsistency(const MetaRegionBBVect &MetaRegions,
                           const std::set<EdgeDescriptor> &Backedges) {
  bool ComparisonState = true;
  for (const MetaRegionBB &MetaRegion : MetaRegions) {
    for (EdgeDescriptor Backedge : Backedges) {
      BasicBlockNodeBB *Source = Backedge.first;
      BasicBlockNodeBB *Target = Backedge.second;
      if (MetaRegion.containsNode(Source)) {
        if ((not MetaRegion.containsNode(Source))
            or (not MetaRegion.containsNode(Source))) {
          ComparisonState = false;
        }
        revng_assert(MetaRegion.containsNode(Source));
        revng_assert(MetaRegion.containsNode(Target));
      }
    }
  }

  return ComparisonState;
}

static void
computeParents(MetaRegionBBVect &MetaRegions, MetaRegionBB *RootMetaRegion) {
  for (MetaRegionBB &MetaRegion1 : MetaRegions) {
    bool ParentFound = false;
    for (MetaRegionBB &MetaRegion2 : MetaRegions) {
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

static MetaRegionBBPtrVect applyPartialOrder(MetaRegionBBVect &V) {
  MetaRegionBBPtrVect OrderedVector;
  std::set<MetaRegionBB *> Processed;

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

static bool alreadyInMetaregion(MetaRegionBBVect &V, BasicBlockNodeBB *N) {

  // Scan all the metaregions and check if a node is already contained in one of
  // them
  for (MetaRegionBB &Region : V) {
    if (Region.containsNode(N)) {
      return true;
    }
  }

  return false;
}

static MetaRegionBBVect
createMetaRegions(const std::set<EdgeDescriptor> &Backedges) {
  std::map<BasicBlockNodeBB *, std::set<BasicBlockNodeBB *>> AdditionalSCSNodes;
  std::vector<std::pair<BasicBlockNodeBB *, std::set<BasicBlockNodeBB *>>>
    Regions;

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
    BasicBlockNodeBB *Head = Region.first;
    std::set<BasicBlockNodeBB *> &Nodes = Region.second;
    std::set<BasicBlockNodeBB *> AdditionalNodes;
    std::set<BasicBlockNodeBB *> OldNodes;
    do {
      OldNodes = Nodes;
      for (BasicBlockNodeBB *Node : Nodes) {
        if ((Node != Head) and (AdditionalSCSNodes.count(Node) != 0)) {
          CombLogger << "Adding additional nodes for region with head: ";
          CombLogger << Head->getNameStr();
          CombLogger << " and relative to node: ";
          CombLogger << Node->getNameStr() << "\n";
          AdditionalNodes.insert(AdditionalSCSNodes[Node].begin(),
                                 AdditionalSCSNodes[Node].end());
        }
      }
      Nodes.insert(AdditionalNodes.begin(), AdditionalNodes.end());
      AdditionalNodes.clear();
    } while (Nodes != OldNodes);
  }

  MetaRegionBBVect MetaRegions;
  int SCSIndex = 1;
  for (size_t I = 0; I < Regions.size(); ++I) {
    auto &SCS = Regions[I].second;
    MetaRegions.push_back(MetaRegionBB(SCSIndex, SCS, true));
    SCSIndex++;
  }
  return MetaRegions;
}

static void
removeFromRPOT(std::vector<BasicBlockNodeBB *> &RPOT, BasicBlockNodeBB *Node) {

  RPOT.erase(std::remove_if(RPOT.begin(),
                            RPOT.end(),
                            [Node](BasicBlockNodeBB *N) {
                              if (N == Node) {
                                return true;
                              }
                              return false;
                            }),
             RPOT.end());
}

char RestructureCFG::ID = 0;
static RegisterPass<RestructureCFG> X("restructure-cfg",
                                      "Apply RegionCFG restructuring "
                                      "transformation",
                                      true,
                                      true);

static cl::opt<std::string> OutputPath("restructure-metrics-output-dir",
                                       desc("Restructure metrics dir"),
                                       value_desc("restructure-dir"),
                                       cat(MainCategory));

ASTTree &RestructureCFG::getAST() {
  return RootCFG.getAST();
}

bool RestructureCFG::runOnFunction(Function &F) {

  DuplicationCounter = 0;
  UntangleTentativeCounter = 0;
  UntanglePerformedCounter = 0;

  // Analyze only isolated functions.
  if (!F.getName().startswith("bb.")
      or F.getName().startswith("bb.quotearg_buffer_restyled")
      or F.getName().startswith("bb.printf_parse")
      or F.getName().startswith("bb.printf_core")
      or F.getName().startswith("bb._Unwind_VRS_Pop")
      or F.getName().startswith("bb.main")
      or F.getName().startswith("bb.vasnprintf")) {
    return false;
  }

  // If we passed the `-single-decompilation` option to the command line, skip
  // decompilation for all the functions that are not the selected one.
  if (TargetFunction.size() != 0) {
    if (!F.getName().equals(TargetFunction.c_str())) {
      return false;
    }
  }

  // Clear graph object from the previous pass.
  RootCFG = RegionCFG<BasicBlock *>();

  // Set names of the CFG region
  RootCFG.setFunctionName(F.getName());
  RootCFG.setRegionName("root");

  // Initialize the RegionCFG object
  RootCFG.initialize(&F);

  // Dump the function name.
  if (CombLogger.isEnabled()) {
    CombLogger << "Analyzing function: " << F.getName() << "\n";
  }

  // Dump the object in .dot format if debug mode is activated.
  if (CombLogger.isEnabled()) {
    RootCFG.dumpDotOnFile("dots", F.getName(), "begin");
  }

  // Identify SCS regions.
  std::set<EdgeDescriptor> Backedges = getBackedges(RootCFG);
  if (CombLogger.isEnabled()) {
    CombLogger << "Backedges in the graph:\n";
    for (auto &Backedge : Backedges) {
      CombLogger << Backedge.first->getNameStr() << " -> "
                 << Backedge.second->getNameStr() << "\n";
    }
  }

  // Insert a dummy node for each retrating node.
  for (EdgeDescriptor Backedge : Backedges) {
    BasicBlockNodeBB *OriginalTarget = Backedge.second;
    BasicBlockNodeBB *Dummy = RootCFG.addArtificialNode();
    moveEdgeTarget(Backedge, Dummy);
    addEdge(EdgeDescriptor(Dummy, OriginalTarget));
  }
  Backedges = getBackedges(RootCFG);

  // Check that the source node of each retreating edge is a dummy node.
  for (EdgeDescriptor Backedge : Backedges) {
    revng_assert(Backedge.first->isEmpty());
  }

  // Create meta regions
  MetaRegionBBVect MetaRegions = createMetaRegions(Backedges);

  // Temporary map where to store the corrispondence between the backedge and
  // the SCS it gives origin to.
  // HACK: this should be done at the same time of the metaregion creation.
  unsigned MetaRegionIndex = 0;
  std::map<EdgeDescriptor, MetaRegionBB *> BackedgeMetaRegionMap;
  for (EdgeDescriptor Backedge : Backedges) {
    BackedgeMetaRegionMap[Backedge] = &MetaRegions.at(MetaRegionIndex);
    MetaRegionIndex++;
  }

  // Print gross metaregions.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after nothing:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      CombLogger << "With index " << Meta.getIndex() << "\n";
      CombLogger << "With size " << Meta.nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }
  }

  // Simplify SCS if they contain an edge which goes outside the scope of the
  // current region.
  simplifySCSAbnormalRetreating(MetaRegions, Backedges, BackedgeMetaRegionMap);

  // Check consitency of metaregions simplified above.
  revng_assert(checkMetaregionConsistency(MetaRegions, Backedges));

  // Print SCS after first simplification.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after first simplification:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      CombLogger << "With index " << Meta.getIndex() << "\n";
      CombLogger << "With size " << Meta.nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }
  }

  // Simplify SCS in a fixed-point fashion.
  simplifySCS(MetaRegions);

  // Check consitency of metaregions simplified above
  revng_assert(checkMetaregionConsistency(MetaRegions, Backedges));

  // Print SCS after second simplification.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after second simplification:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      CombLogger << "With index " << Meta.getIndex() << "\n";
      CombLogger << "With size " << Meta.nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
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
      CombLogger << "With index " << Meta.getIndex() << "\n";
      CombLogger << "With size " << Meta.nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }
  }

  // Compute parent relations for the identified SCSs.
  std::set<BasicBlockNodeBB *> Empty;
  MetaRegionBB RootMetaRegion(0, Empty);
  computeParents(MetaRegions, &RootMetaRegion);

  // Print metaregions after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions parent relationship:\n";
    for (auto &Meta : MetaRegions) {
      CombLogger << "\n";
      CombLogger << &Meta << "\n";
      CombLogger << "With index " << Meta.getIndex() << "\n";
      CombLogger << "With size " << Meta.nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta.getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
      CombLogger << "Has parent: " << Meta.getParent() << "\n";
    }
  }

  // Find an ordering for the metaregions that satisfies the inclusion
  // relationship. We create a new "shadow" vector containing only pointers to
  // the "real" metaregions.
  MetaRegionBBPtrVect OrderedMetaRegions = applyPartialOrder(MetaRegions);

  // Print metaregions after ordering.
  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Metaregions after partial ordering:\n";
    for (auto *Meta : OrderedMetaRegions) {
      CombLogger << "\n";
      CombLogger << Meta << "\n";
      CombLogger << "With index " << Meta->getIndex() << "\n";
      CombLogger << "With size " << Meta->nodes_size() << "\n";
      CombLogger << "Is composed of nodes:\n";
      auto &Nodes = Meta->getNodes();
      for (auto *Node : Nodes) {
        CombLogger << Node->getNameStr() << "\n";
      }
      CombLogger << "Has parent: " << Meta->getParent() << "\n";
      CombLogger << "Is SCS: " << Meta->isSCS() << "\n";
    }
  }

  ReversePostOrderTraversal<BasicBlockNodeBB *> ORPOT(&RootCFG.getEntryNode());

  // Create a std::vector from the reverse post order (we will later need
  // the removal operation)
  std::vector<BasicBlockNodeBB *> RPOT;
  for (BasicBlockNodeBB *BN : ORPOT) {
    RPOT.push_back(BN);
  }

  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Reverse post order is:\n";
    for (BasicBlockNodeBB *BN : RPOT) {
      CombLogger << BN->getNameStr() << "\n";
    }
    CombLogger << "Reverse post order end\n";
  }

  DominatorTreeBase<BasicBlockNodeBB, false> DT;
  DT.recalculate(RootCFG);

  DominatorTreeBase<BasicBlockNodeBB, true> PDT;
  PDT.recalculate(RootCFG);

  // Reserve enough space for all the OrderedMetaRegions.
  // The following algorithms stores pointers to the elements of this vector, so
  // we need to make sure that no reallocation happens.
  std::vector<RegionCFG<BasicBlock *>> Regions(OrderedMetaRegions.size());

  for (MetaRegionBB *Meta : OrderedMetaRegions) {
    if (CombLogger.isEnabled()) {
      CombLogger << "\nAnalyzing region: " << Meta->getIndex() << "\n";
    }

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

    // Identify all the abnormal retreating edges in a SCS.
    std::set<EdgeDescriptor> Retreatings;
    std::set<BasicBlockNodeBB *> RetreatingTargets;
    for (EdgeDescriptor Backedge : Backedges) {
      if (Meta->containsNode(Backedge.first)) {

        // Check that the target of the retreating edge falls inside the current
        // SCS.
        revng_assert(Meta->containsNode(Backedge.second));

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

    // We need to update the backedges list removing the edges which have been
    // considered as retreatings of the SCS under analysis.
    for (EdgeDescriptor Retreating : Retreatings) {
      revng_assert(Backedges.count(Retreating) == 1);
      Backedges.erase(Retreating);
    }

    // Always take the fist node in RPOT which is a retreating target as entry,
    // candidate.
    BasicBlockNodeBB *FirstCandidate = nullptr;
    for (BasicBlockNodeBB *BN : RPOT) {
      if (Meta->containsNode(BN) == true and RetreatingTargets.count(BN) == 1) {
        FirstCandidate = BN;
        break;
      }
    }

    revng_assert(FirstCandidate != nullptr);

    // Print out the name of the node that has been selected as head of the
    // region
    if (CombLogger.isEnabled()) {
      CombLogger << "Elected head is: " << FirstCandidate->getNameStr() << "\n";
    }

    bool NewHeadNeeded = false;
    for (BasicBlockNodeBB *Node : RetreatingTargets) {
      if (Node != FirstCandidate) {
        NewHeadNeeded = true;
      }
    }
    if (CombLogger.isEnabled()) {
      CombLogger << "New head needed: " << NewHeadNeeded << "\n";
    }

    BasicBlockNodeBB *Head;
    if (NewHeadNeeded) {
      revng_assert(RetreatingTargets.size() > 1);

      // Create the dispatcher.
      Head = RootCFG.addEntryDispatcher();
      Meta->insertNode(Head);

      // For each target of the dispatcher add the edge and add it in the map.
      std::map<BasicBlockNodeBB *, unsigned> RetreatingIdxMap;
      unsigned Idx = 0;
      for (BasicBlockNodeBB *Target : RetreatingTargets) {
        RetreatingIdxMap[Target] = Idx;
        Idx++;
        Head->addSuccessor(Target);
        Target->addPredecessor(Head);
      }

      // Check that we inserted the correct number of edges.
      revng_assert(Idx == RetreatingTargets.size());

      for (EdgeDescriptor R : Retreatings) {
        BasicBlockNodeBB *OriginalSource = R.first;

        // If the original source is a set node, move it after the entry
        // dispatcher.
        if (OriginalSource->isSet()) {
          BasicBlockNodeBB *OldSetNode = OriginalSource;
          Idx = RetreatingIdxMap[R.second];
          revng_assert(OldSetNode->predecessor_size() == 1);
          BasicBlockNodeBB *Predecessor = OldSetNode->getPredecessorI(0);
          auto *SetNode = RootCFG.addSetStateNode(Idx, OldSetNode->getName());
          Meta->insertNode(SetNode);
          moveEdgeTarget(EdgeDescriptor(Predecessor, OldSetNode), Head);
        } else {
          Idx = RetreatingIdxMap[R.second];
          auto *SetNode = RootCFG.addSetStateNode(Idx, R.second->getName());
          Meta->insertNode(SetNode);
          moveEdgeTarget(EdgeDescriptor(R.first, R.second), SetNode);
          addEdge(EdgeDescriptor(SetNode, Head));
        }
      }

      // Move the incoming edge from the old head to new one.
      std::vector<BasicBlockNodeBB *> Predecessors;
      for (BasicBlockNodeBB *Predecessor : FirstCandidate->predecessors())
        Predecessors.push_back(Predecessor);

      for (BasicBlockNodeBB *Predecessor : Predecessors) {
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
    std::set<BasicBlockNodeBB *> Successors = Meta->getSuccessors();

    if (CombLogger.isEnabled()) {
      CombLogger << "Region successors are:\n";
      for (BasicBlockNodeBB *Node : Successors) {
        CombLogger << Node->getNameStr() << "\n";
      }
    }

    bool AnotherIteration = true;
    while (AnotherIteration and Successors.size() > 1) {
      AnotherIteration = false;
      std::set<EdgeDescriptor> OutgoingEdges = Meta->getOutEdges();

      std::vector<BasicBlockNodeBB *> Frontiers;
      std::map<BasicBlockNodeBB *, pair<BasicBlockNodeBB *, BasicBlockNodeBB *>>
        EdgeExtremal;

      for (EdgeDescriptor Edge : OutgoingEdges) {
        BasicBlockNodeBB *Frontier = RootCFG.addArtificialNode();
        BasicBlockNodeBB *OldSource = Edge.first;
        BasicBlockNodeBB *OldTarget = Edge.second;
        EdgeExtremal[Frontier] = make_pair(OldSource, OldTarget);
        moveEdgeTarget(Edge, Frontier);
        addEdge(EdgeDescriptor(Frontier, OldTarget));
        Meta->insertNode(Frontier);
        Frontiers.push_back(Frontier);
      }

      DT.recalculate(RootCFG);
      for (BasicBlockNodeBB *Frontier : Frontiers) {
        for (BasicBlockNodeBB *Successor : Successors) {
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
      for (BasicBlockNodeBB *Frontier : Frontiers) {
        BasicBlockNodeBB *OriginalSource = EdgeExtremal[Frontier].first;
        BasicBlockNodeBB *OriginalTarget = EdgeExtremal[Frontier].second;
        moveEdgeTarget({ OriginalSource, Frontier }, OriginalTarget);
        RootCFG.removeNode(Frontier);
        Meta->removeNode(Frontier);
      }

      Successors = Meta->getSuccessors();
    }

    // First Iteration outlining.
    // Clone all the nodes of the SCS except for the head.
    std::map<BasicBlockNodeBB *, BasicBlockNodeBB *> ClonedMap;
    std::vector<BasicBlockNodeBB *> OutlinedNodes;
    for (BasicBlockNodeBB *Node : Meta->nodes()) {
      if (Node != Head) {
        BasicBlockNodeBB *Clone = RootCFG.cloneNode(*Node);
        Clone->setName(Node->getName().str() + " outlined");
        ClonedMap[Node] = Clone;

        // Add the nodes to the additional vector
        OutlinedNodes.push_back(Clone);
      }
    }

    // Restore edges between cloned nodes.
    for (BasicBlockNodeBB *Node : Meta->nodes()) {
      if (Node != Head) {

        // Handle outgoing edges from SCS nodes.
        for (BasicBlockNodeBB *Successor : Node->successors()) {
          revng_assert(!Backedges.count(EdgeDescriptor(Node, Successor)));
          using ED = EdgeDescriptor;
          if (Meta->containsNode(Successor)) {
            // Handle edges pointing inside the SCS.
            if (Successor == Head) {
              // Retreating edges should point to the new head.
              addEdge(ED(ClonedMap.at(Node), Head));
            } else {
              // Other edges should be restored between cloned nodes.
              addEdge(ED(ClonedMap.at(Node), ClonedMap.at(Successor)));
            }
          } else {
            // Edges exiting from the SCS should go to the right target.
            addEdge(ED(ClonedMap.at(Node), Successor));
          }
        }

        // We need this temporary vector to avoid invalidating iterators.
        std::vector<BasicBlockNodeBB *> Predecessors;
        for (BasicBlockNodeBB *Predecessor : Node->predecessors()) {
          Predecessors.push_back(Predecessor);
        }
        for (BasicBlockNodeBB *Predecessor : Predecessors) {
          if (!Meta->containsNode(Predecessor)) {
            // Is the edge we are moving a backedge ?.
            if (CombLogger.isEnabled()) {
              CombLogger << "Index region: " << Meta->getIndex() << "\n";
              CombLogger << "Backedge that we would insert: "
                         << Predecessor->getNameStr() << " -> "
                         << Node->getNameStr() << "\n";
            }

            // Are we moving a backedge with the first iteration outlining?
            revng_assert(!Backedges.count(EdgeDescriptor(Predecessor, Node)));

            moveEdgeTarget(EdgeDescriptor(Predecessor, Node),
                           ClonedMap.at(Node));
          }
        }
      }
    }

    // Vector which contains the additional set nodes that set the default value
    // for the entry dispatcher.
    std::vector<BasicBlockNodeBB *> DefaultEntrySet;

    // Default set node for entry dispatcher.
    if (NewHeadNeeded) {
      revng_assert(Head->isDispatcher());
      std::set<BasicBlockNodeBB *> SetCandidates;
      for (BasicBlockNodeBB *Pred : Head->predecessors()) {
        if (not Pred->isSet()) {
          SetCandidates.insert(Pred);
        }
      }
      unsigned long Value = RetreatingTargets.size() - 1;
      for (BasicBlockNodeBB *Pred : SetCandidates) {
        BasicBlockNodeBB *Set = RootCFG.addSetStateNode(Value, Head->getName());
        DefaultEntrySet.push_back(Set);
        moveEdgeTarget(EdgeDescriptor(Pred, Head), Set);
        addEdge(EdgeDescriptor(Set, Head));

        // HACK: Consider using a multimap.
        //
        // Update the backedges set. Basically, when we place the default set
        // node in case of an entry dispatcher, we need to take care to verify
        // if the edge we are "moving" (inserting the set node before it) is a
        // backedge, and in case update the information regarding the backedges
        // present in the graph accordingly (the backedge becomes the edge
        // departing from the set node).
        bool UpdatedBackedges = true;
        while (UpdatedBackedges) {
          UpdatedBackedges = false;
          for (EdgeDescriptor Backedge : Backedges) {
            BasicBlockNodeBB *Source = Backedge.first;
            if (Source == Pred) {
              Backedges.erase(Backedge);
              Backedges.insert(EdgeDescriptor(Set, Head));
              UpdatedBackedges = true;
              break;
            }
          }
        }
      }
    }

    // Exit dispatcher creation.
    // TODO: Factorize this out together with the head dispatcher creation.
    bool NewExitNeeded = false;
    BasicBlockNodeBB *Exit = nullptr;
    std::vector<BasicBlockNodeBB *> ExitDispatcherNodes;
    if (Successors.size() > 1) {
      NewExitNeeded = true;
    }
    if (CombLogger.isEnabled()) {
      CombLogger << "New exit needed: " << NewExitNeeded << "\n";
    }

    if (NewExitNeeded) {
      revng_assert(Successors.size() > 1);

      // Create the dispatcher.
      Exit = RootCFG.addExitDispatcher();
      ExitDispatcherNodes.push_back(Exit);

      // For each target of the dispatcher add the edge and add it in the map.
      std::map<BasicBlockNodeBB *, unsigned> SuccessorsIdxMap;
      unsigned Idx = 0;
      for (BasicBlockNodeBB *Successor : Successors) {
        SuccessorsIdxMap[Successor] = Idx;
        Idx++;

        Exit->addSuccessor(Successor);
        Successor->addPredecessor(Exit);
      }

      // Check that we inserted the correct number of edges.
      revng_assert(Idx == Successors.size());

      std::set<EdgeDescriptor> OutEdges = Meta->getOutEdges();
      for (EdgeDescriptor Edge : OutEdges) {
        Idx = SuccessorsIdxMap.at(Edge.second);
        auto *IdxSetNode = RootCFG.addSetStateNode(Idx, Edge.second->getName());
        Meta->insertNode(IdxSetNode);
        moveEdgeTarget(EdgeDescriptor(Edge.first, Edge.second), IdxSetNode);
        addEdge(EdgeDescriptor(IdxSetNode, Edge.second));

        // We should not be adding new backedges.
        revng_assert(Backedges.count(Edge) == 0);
      }
      if (CombLogger.isEnabled()) {
        CombLogger << "New exit name is: " << Exit->getNameStr() << "\n";
      }
    }

    // Collapse Region.
    // Create a new RegionCFG object for representing the collapsed region and
    // populate it with the internal nodes.
    Regions.push_back(RegionCFG<BasicBlock *>());
    RegionCFG<BasicBlock *> &CollapsedGraph = Regions.back();
    RegionCFG<BasicBlock *>::BBNodeMap SubstitutionMap{};
    CollapsedGraph.setFunctionName(F.getName());
    CollapsedGraph.setRegionName(std::to_string(Meta->getIndex()));
    revng_assert(Head != nullptr);

    // Create the collapsed node in the outer region.
    BasicBlockNodeBB *Collapsed = RootCFG.createCollapsedNode(&CollapsedGraph);

    // Hack: we should use a std::multimap here, so that we can update the
    // target of the edgedescriptor in place without having to remove and insert
    // from the set and invalidating iterators.
    //
    // Update the backedges set, checking that if a backedge of an outer region
    // pointed to a node that now has been collapsed, now should point to the
    // collapsed node, and that does not exists at this point a backedge which
    // has as source a node that will be collapsed.
    bool UpdatedBackedges = true;
    while (UpdatedBackedges) {
      UpdatedBackedges = false;
      for (EdgeDescriptor Backedge : Backedges) {
        BasicBlockNodeBB *Source = Backedge.first;
        BasicBlockNodeBB *Target = Backedge.second;
        revng_assert(!Meta->containsNode(Source));
        if (Meta->containsNode(Target)) {
          revng_assert(Target == Head);
          Backedges.erase(Backedge);
          Backedges.insert(EdgeDescriptor(Source, Collapsed));
          UpdatedBackedges = true;
          break;
        }
      }
    }

    CollapsedGraph.insertBulkNodes(Meta->getNodes(), Head, SubstitutionMap);

    // Connect the break and continue nodes with the necessary edges (we create
    // a new break/continue node for each outgoing or retreating edge).
    CollapsedGraph.connectContinueNode();
    std::set<EdgeDescriptor> OutgoingEdges = Meta->getOutEdges();
    CollapsedGraph.connectBreakNode(OutgoingEdges, SubstitutionMap);

    // Connect the old incoming edges to the collapsed node.
    std::set<EdgeDescriptor> IncomingEdges = Meta->getInEdges();
    for (EdgeDescriptor Edge : IncomingEdges) {
      BasicBlockNodeBB *OldSource = Edge.first;
      revng_assert(Edge.second == Head);

      // Check if the old edge was a backedge edge, and in case update the
      // information about backedges accordingly.
      if (Backedges.count(Edge) == 1) {
        Backedges.erase(Edge);
        Backedges.insert(EdgeDescriptor(OldSource, Collapsed));
      }

      moveEdgeTarget(Edge, Collapsed);
    }

    // Connect the outgoing edges to the collapsed node.
    if (NewExitNeeded) {
      revng_assert(Exit != nullptr);
      addEdge(EdgeDescriptor(Collapsed, Exit));
    } else {

      // Double check that we have at most a single successor
      revng_assert(Successors.size() <= 1);
      if (Successors.size() == 1) {

        // Connect the collapsed node to the unique successor
        BasicBlockNodeBB *Successor = *Successors.begin();
        addEdge(EdgeDescriptor(Collapsed, Successor));
      }
    }

    // Remove collapsed nodes from the outer region.
    for (BasicBlockNodeBB *Node : Meta->nodes()) {
      if (CombLogger.isEnabled()) {
        CombLogger << "Removing from main graph node :" << Node->getNameStr()
                   << "\n";
      }
      RootCFG.removeNode(Node);
      removeFromRPOT(RPOT, Node);
    }

    // Substitute in the other SCSs the nodes of the current SCS with the
    // collapsed node and the exit dispatcher structure.
    for (MetaRegionBB *OtherMeta : OrderedMetaRegions) {
      if (OtherMeta != Meta) {
        OtherMeta->updateNodes(Meta->getNodes(),
                               Collapsed,
                               ExitDispatcherNodes,
                               DefaultEntrySet,
                               OutlinedNodes);
      }
    }

    // Replace the pointers inside SCS.
    Meta->replaceNodes(CollapsedGraph.getNodes());

    // Remove useless nodes inside the SCS (like dandling break/continue)
    CollapsedGraph.removeNotReachables(OrderedMetaRegions);

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
    RootCFG.removeNotReachables(OrderedMetaRegions);

    // Check that the newly created collapsed region is acyclic.
    revng_assert(CollapsedGraph.isDAG());
  }

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph before final purge\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "Final-before-purge");
  }

  // Remove not reachables nodes from the main final graph.
  RootCFG.removeNotReachables(OrderedMetaRegions);

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

  // Check that the root region is acyclic at this point.
  revng_assert(RootCFG.isDAG());

  // Compute the initial weight of the CFG.
  unsigned InitialWeight = 0;
  for (BasicBlockNodeBB *BBNode : RootCFG.nodes()) {
    InitialWeight += BBNode->getWeight();
  }

  // Invoke the AST generation for the root region.
  RootCFG.generateAst();

  // Serialize final AST on file
  if (CombLogger.isEnabled()) {
    RootCFG.getAST().dumpOnFile("ast", F.getName(), "Final");
  }

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

  // Collect the number of cloned nodes introduced by the comb for a single
  // `llvm::BasicBlock`, information which is needed later in the
  // `MarkForSerialization` pass.
  //
  // Collect also the final weight of the CFG.
  unsigned FinalWeight = 0;
  for (BasicBlockNodeBB *BBNode : RootCFG.nodes()) {
    BasicBlock *BB = BBNode->getOriginalNode();
    if (BBNode->isCode()) {
      revng_assert(BB != nullptr);
      NDuplicates[BB] += 1;
      // if (NDuplicates[BB] > 1)
      // DuplicationCounter += 1;
    } else {
      revng_assert(BB == nullptr);
    }

    // Collect the weight of the node.
    FinalWeight += BBNode->getWeight();
  }

  // Serialize final AST after flattening on file
  if (CombLogger.isEnabled()) {
    RootCFG.getAST().dumpOnFile("ast", F.getName(), "Final-after-flattening");
  }

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph after Flattening\n";
    RootCFG.dumpDotOnFile("dots", F.getName(), "final-after-flattening");
  }

  // Compute the increase in weight.
  float Increase = float(FinalWeight) / float(InitialWeight);

  // Serialize the collected metrics in the outputfile.
  if (OutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    const char *FunctionName = F.getName().data();
    std::ostream &OutputStream = pathToStream(OutputPath + "/" + FunctionName,
                                              Output);
    OutputStream << "function,"
                    "duplications,percentage,tuntangle,puntangle,iweight\n";
    OutputStream << F.getName().data() << "," << DuplicationCounter << ","
                 << Increase << "," << UntangleTentativeCounter << ","
                 << UntanglePerformedCounter << "," << InitialWeight << "\n";
  }

  return false;
}
