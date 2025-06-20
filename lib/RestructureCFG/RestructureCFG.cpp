//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <limits>
#include <sstream>
#include <utility>

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/RestructureCFG/ASTTree.h"
#include "revng/RestructureCFG/BasicBlockNodeImpl.h"
#include "revng/RestructureCFG/GenerateAst.h"
#include "revng/RestructureCFG/MetaRegionBB.h"
#include "revng/RestructureCFG/RegionCFGTreeBB.h"
#include "revng/RestructureCFG/RestructureCFG.h"
#include "revng/RestructureCFG/Utils.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace llvm::cl;

using std::pair;
using std::string;

// TODO: Move the initialization of the logger here from "Utils.h"
// Debug logger.
Logger<> CombLogger("restructure");
Logger<> LogShortestPath("restructure-shortest-path");

// EdgeDescriptor is a handy way to create and manipulate edges on the
// RegionCFG.
using BasicBlockNodeBB = BasicBlockNode<BasicBlock *>;
using EdgeDescriptor = std::pair<BasicBlockNodeBB *, BasicBlockNodeBB *>;

// Explicit instantiation of template classes `Metaregion` and `RegionCFG`.
template class MetaRegion<BasicBlock *>;
template class RegionCFG<BasicBlock *>;
using MetaRegionBB = MetaRegion<BasicBlock *>;
using MetaRegionBBVect = std::vector<MetaRegionBB>;
using MetaRegionBBPtrVect = std::vector<MetaRegionBB *>;
using BackedgeMetaRegionMap = std::map<EdgeDescriptor, MetaRegionBB *>;

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
                           const llvm::SmallDenseSet<EdgeDescriptor> &Backedges,
                           BackedgeMetaRegionMap &BackedgeMetaRegionMap,
                           std::set<MetaRegionBB *> &BlacklistedMetaregions) {
  for (auto RegionIt = MetaRegions.begin(); RegionIt != MetaRegions.end();
       RegionIt++) {
    MetaRegionBB &Region = *RegionIt;

    // Do not re-analyze blacklisted metaregions.
    if (!BlacklistedMetaregions.contains(&Region)) {

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
                              const llvm::SmallDenseSet<EdgeDescriptor>
                                &Backedges) {

  // Temporary map where to store the correspondence between the backedge and
  // the SCS it gives origin to.
  // HACK: this should be done at the same time of the metaregion creation.
  unsigned MetaRegionIndex = 0;
  std::map<EdgeDescriptor, MetaRegionBB *> BackedgeMetaRegionMap;
  for (EdgeDescriptor Backedge : Backedges) {
    BackedgeMetaRegionMap[Backedge] = &MetaRegions.at(MetaRegionIndex);
    MetaRegionIndex++;
  }

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

static bool checkMetaregionConsistency(const MetaRegionBBVect &MetaRegions,
                                       const llvm::SmallDenseSet<EdgeDescriptor>
                                         &Backedges) {
  bool ComparisonState = true;
  for (const MetaRegionBB &MetaRegion : MetaRegions) {
    for (EdgeDescriptor Backedge : Backedges) {
      BasicBlockNodeBB *Source = Backedge.first;
      BasicBlockNodeBB *Target = Backedge.second;
      bool HasSource = MetaRegion.containsNode(Source);
      bool HasTarget = MetaRegion.containsNode(Target);
      revng_assert(HasSource == HasTarget);
      if (HasSource != HasTarget) {
        ComparisonState = false;
      }
    }
  }

  return ComparisonState;
}

static void computeParents(MetaRegionBBVect &MetaRegions) {
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

      MetaRegion1.setParent(nullptr);
    }
  }
}

static MetaRegionBBPtrVect applyPartialOrder(MetaRegionBBVect &V) {
  MetaRegionBBPtrVect OrderedVector;
  std::set<MetaRegionBB *> Processed;

  while (V.size() != Processed.size()) {
    for (auto RegionIt1 = V.begin(); RegionIt1 != V.end(); RegionIt1++) {
      if (!Processed.contains(&*RegionIt1)) {
        bool FoundParent = false;
        for (auto RegionIt2 = V.begin(); RegionIt2 != V.end(); RegionIt2++) {
          if ((RegionIt1 != RegionIt2) and !Processed.contains(&*RegionIt2)) {
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
  for (MetaRegionBB &Region : V)
    if (Region.containsNode(N))
      return true;
  return false;
}

static MetaRegionBBVect
createMetaRegions(const llvm::SmallDenseSet<EdgeDescriptor> &Backedges) {
  std::map<BasicBlockNodeBB *, std::set<BasicBlockNodeBB *>> AdditionalSCSNodes;
  std::vector<std::pair<BasicBlockNodeBB *, std::set<BasicBlockNodeBB *>>>
    Regions;

  for (auto &Backedge : Backedges) {

    // Convert the `llvm::SmallSetVector` generated by the `nodesBetween` to a
    // `std::set`, whose ordering properties are necessary for the following of
    // the restructuring algorithm
    auto SCSNodesSmall = nodesBetween(Backedge.second, Backedge.first);
    std::set<BasicBlockNodeBB *> SCSNodes;
    SCSNodes.insert(SCSNodesSmall.begin(), SCSNodesSmall.end());
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
        if ((Node != Head) and (AdditionalSCSNodes.contains(Node))) {
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

static cl::opt<std::string> MetricsOutputPath("restructure-metrics-output-dir",
                                              desc("Restructure metrics dir"),
                                              value_desc("restructure-dir"),
                                              cat(MainCategory));

static void LogMetaRegions(const MetaRegionBBPtrVect &MetaRegions,
                           const std::string &HeaderMsg) {
  if (CombLogger.isEnabled()) {
    CombLogger << '\n';
    CombLogger << HeaderMsg << '\n';
    for (const MetaRegionBB *Meta : MetaRegions) {
      CombLogger << '\n';
      CombLogger << Meta << '\n';
      CombLogger << "With index " << Meta->getIndex() << '\n';
      CombLogger << "With size " << Meta->nodes_size() << '\n';
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Meta->nodes())
        CombLogger << Node->getNameStr() << '\n';
      CombLogger << "Is SCS: " << Meta->isSCS() << '\n';
      CombLogger << "Has parent: ";
      if (Meta->getParent())
        CombLogger << Meta->getParent();
      else
        CombLogger << "nullptr";
      CombLogger << '\n';
    }
  }
}

static void LogMetaRegions(const MetaRegionBBVect &MetaRegions,
                           const std::string &HeaderMsg) {
  if (CombLogger.isEnabled()) {
    CombLogger << '\n';
    CombLogger << HeaderMsg << '\n';
    for (const MetaRegionBB &Meta : MetaRegions) {
      CombLogger << '\n';
      CombLogger << &Meta << '\n';
      CombLogger << "With index " << Meta.getIndex() << '\n';
      CombLogger << "With size " << Meta.nodes_size() << '\n';
      CombLogger << "Is composed of nodes:\n";
      for (auto *Node : Meta.nodes())
        CombLogger << Node->getNameStr() << '\n';
      CombLogger << "Is SCS: " << Meta.isSCS() << '\n';
      CombLogger << "Has parent: ";
      if (Meta.getParent())
        CombLogger << Meta.getParent();
      else
        CombLogger << "nullptr";
      CombLogger << '\n';
    }
  }
}

static debug_function void LogMetaRegions(const MetaRegionBBVect &MetaRegions,
                                          const char *HeaderMsg) {
  LogMetaRegions(MetaRegions, std::string(HeaderMsg));
}

static std::map<BasicBlockNodeBB *, size_t>
getCandidateEntries(MetaRegionBB *Meta) {
  std::map<BasicBlockNodeBB *, size_t> Result;
  std::set<EdgeDescriptor> InEdges = Meta->getInEdges();
  for (const auto &[Src, Tgt] : InEdges)
    ++Result[Tgt];
  return Result;
}

// Function to compute the most nested regions between the ones passed in the
// parameter `SmallSet`. The assumption is that the candidate `MetaRegion`s
// passed as parameters all lie on a single nesting derivation line, i.e., the
// need to be all descendant of one another.
static MetaRegionBB *
mostNestedRegion(llvm::SmallSet<MetaRegionBB *, 4> &MetaRegions) {

  // Select the most nested `MetaRegion`
  MetaRegionBB *MaxMetaRegion = nullptr;
  size_t MaxLevel = 0;
  for (MetaRegionBB *Meta : MetaRegions) {

    // If we encounter the `root` `MetaRegion`, we do not proceed with the body
    // of the loop, since the initialization value already represents the `root`
    if (Meta == nullptr) {
      continue;
    }

    // Compute the nesting level for each `MetaRegion`
    MetaRegionBB *UpwardMeta = Meta;
    size_t Level = 0;
    while (UpwardMeta != nullptr) {
      UpwardMeta = UpwardMeta->getParent();
      Level++;
    }

    // Due to the initial assumption that all the input `MetaRegion`s lie on a
    // single nesting tree in the `MetaRegion` containement tree, we should
    // never encounter the same level twice
    revng_assert(Level != MaxLevel);

    // If the level reached at this iteration is greater than what found
    // previously, we update the value
    if (Level > MaxLevel) {
      MaxMetaRegion = Meta;
      MaxLevel = Level;
    }
  }

  return MaxMetaRegion;
}

// Function that computes the most nested `MetaRegion` parent between the
// predecessors of the `Node` input block.
static MetaRegionBB *computePredecessorsParent(MetaRegionBBPtrVect &MetaRegions,
                                               BasicBlockNodeBB *Node) {

  // Elect the parent `MetaRegion` for each of the `Node` predecessor
  llvm::SmallSet<MetaRegionBB *, 4> PredecessorMetaRegions;
  for (BasicBlockNodeBB *Predecessor : Node->predecessors()) {

    // Collect all the `MetaRegion`s containing the `Predecessor`
    llvm::SmallSet<MetaRegionBB *, 4> ContainingMetaRegions;
    for (MetaRegionBB *Meta : MetaRegions) {
      if (Meta->containsNode(Predecessor)) {
        ContainingMetaRegions.insert(Meta);
      }
    }

    // Elect the most nested `MetaRegion` for each `Predecessor`
    MetaRegionBB *Meta = mostNestedRegion(ContainingMetaRegions);
    PredecessorMetaRegions.insert(Meta);
  }

  // Elect the most nested `MetaRegion` between each one selected from the input
  // `MetaRegion
  return mostNestedRegion(PredecessorMetaRegions);
}

bool restructureCFG(Function &F, ASTTree &AST) {
  revng_log(CombLogger, "restructuring Function: " << F.getName());
  revng_log(CombLogger, "Num basic blocks: " << F.size());

  DuplicationCounter = 0;
  UntangleTentativeCounter = 0;
  UntanglePerformedCounter = 0;

  // Clear graph object from the previous pass.
  RegionCFG<BasicBlock *> RootCFG;

  // Set names of the CFG region
  RootCFG.setFunctionName(F.getName().str());
  RootCFG.setRegionName("root");

  // Initialize the RegionCFG object
  RootCFG.initialize(&F);

  if (CombLogger.isEnabled()) {
    CombLogger << "Analyzing function: " << F.getName() << "\n";
    RootCFG.dumpCFGOnFile(F.getName().str(), "restructure", "initial-state");
  }

  // Identify SCS regions.
  llvm::SmallDenseSet<EdgeDescriptor>
    Backedges = getBackedges(&RootCFG.getEntryNode()).takeSet();
  revng_log(CombLogger, "Initial Backedges in the graph:");
  for (auto &Backedge : Backedges) {
    LoggerIndent Indent(CombLogger);
    revng_log(CombLogger,
              Backedge.first->getNameStr()
                << " -> " << Backedge.second->getNameStr());
  }

  // Insert a dummy node for each retreating node.
  for (EdgeDescriptor Backedge : Backedges) {
    BasicBlockNodeBB *OriginalTarget = Backedge.second;
    BasicBlockNodeBB *Dummy = RootCFG.addArtificialNode();
    moveEdgeTarget(Backedge, Dummy);
    addPlainEdge(EdgeDescriptor(Dummy, OriginalTarget));
  }
  Backedges.clear();
  Backedges = getBackedges(&RootCFG.getEntryNode()).takeSet();

  // Check that the source node of each retreating edge is a dummy node.
  revng_log(CombLogger, "Backedges in the graph after dummy insertion:");
  for (auto &Backedge : Backedges) {
    LoggerIndent Indent(CombLogger);
    revng_log(CombLogger,
              Backedge.first->getNameStr()
                << " -> " << Backedge.second->getNameStr());
    revng_assert(Backedge.first->isEmpty());
  }

  // Create meta regions
  MetaRegionBBVect MetaRegions = createMetaRegions(Backedges);
  LogMetaRegions(MetaRegions, "Metaregions after nothing:");

  // Simplify SCS if they contain an edge which goes outside the scope of the
  // current region.
  simplifySCSAbnormalRetreating(MetaRegions, Backedges);
  LogMetaRegions(MetaRegions, "Metaregions after first simplification:");
  revng_assert(checkMetaregionConsistency(MetaRegions, Backedges));

  // Simplify SCS in a fixed-point fashion.
  simplifySCS(MetaRegions);
  LogMetaRegions(MetaRegions, "Metaregions after second simplification:");
  revng_assert(checkMetaregionConsistency(MetaRegions, Backedges));

  // Sort the Metaregions in increasing number of composing nodes order.
  sortMetaRegions(MetaRegions);
  LogMetaRegions(MetaRegions, "Metaregions after second ordering:");

  // Compute parent relations for the identified SCSs.
  computeParents(MetaRegions);

  // Print metaregions after ordering.
  LogMetaRegions(MetaRegions, "Metaregions parent relationship:");

  // Find an ordering for the metaregions that satisfies the inclusion
  // relationship. We create a new "shadow" vector containing only pointers to
  // the "real" metaregions.
  MetaRegionBBPtrVect OrderedMetaRegions = applyPartialOrder(MetaRegions);

  // Print metaregions after ordering.
  LogMetaRegions(OrderedMetaRegions, "Metaregions after partial ordering:");

  // Create a std::vector from the reverse post order. We cannot just use the
  // regular ReversePostOrderTraversal because later we'll need the removal
  // operation.
  std::vector<BasicBlockNodeBB *> RPOT;
  using RPOTraversal = ReversePostOrderTraversal<BasicBlockNodeBB *>;
  llvm::copy(RPOTraversal{ &RootCFG.getEntryNode() }, std::back_inserter(RPOT));

  if (CombLogger.isEnabled()) {
    CombLogger << "\n";
    CombLogger << "Reverse post order is:\n";
    for (const BasicBlockNodeBB *BN : RPOT)
      CombLogger << BN->getNameStr() << "\n";
    CombLogger << "Reverse post order end\n";
  }

  // Compute shortest path to reach all nodes from Entry.
  // Used later for picking the entry point of each region.
  std::map<BasicBlockNodeBB *, size_t> ShortestPathFromEntry;
  {
    revng_log(LogShortestPath, "Computing ShortestPathFromEntry");
    LoggerIndent Indent(LogShortestPath);
    auto BFSIt = llvm::bf_begin(&RootCFG.getEntryNode());
    auto BFSEnd = llvm::bf_end(&RootCFG.getEntryNode());
    for (; BFSIt != BFSEnd; ++BFSIt) {
      BasicBlockNodeBB *Node = *BFSIt;
      size_t Depth = BFSIt.getLevel();
      revng_log(LogShortestPath, "Node = " << Node);
      auto ShortestIt = ShortestPathFromEntry.lower_bound(Node);
      LoggerIndent MoreIndent(LogShortestPath);
      if (ShortestIt == ShortestPathFromEntry.end()
          or Node < ShortestIt->first) {
        revng_log(LogShortestPath, "New shortest path Depth: " << Depth);
        ShortestPathFromEntry.insert(ShortestIt, { Node, Depth });
      } else {
        revng_log(LogShortestPath,
                  "Known shortest path Depth: " << ShortestIt->second);
        revng_assert(ShortestIt->second <= Depth);
      }
    }
  }

  // Reserve enough space for all the OrderedMetaRegions.
  // The following algorithms stores pointers to the elements of this vector, so
  // we need to make sure that no reallocation happens.
  std::vector<RegionCFG<BasicBlock *>> Regions(OrderedMetaRegions.size());

  for (MetaRegionBB *Meta : OrderedMetaRegions) {
    if (CombLogger.isEnabled()) {
      CombLogger << "\nAnalyzing region: " << Meta->getIndex() << "\n";

      CombLogger << "Which is composed of nodes:\n";
      for (auto *Node : Meta->nodes())
        CombLogger << Node->getNameStr() << "\n";

      CombLogger << "Dumping main graph snapshot before restructuring\n";
      RootCFG.dumpCFGOnFile(F.getName().str(),
                            "restructure",
                            "region-" + std::to_string(Meta->getIndex())
                              + "-outside-before");
      CombLogger.flush();
    }

    // Identify all the abnormal retreating edges in a SCS.
    for (EdgeDescriptor Backedge : llvm::make_early_inc_range(Backedges)) {
      if (Meta->containsNode(Backedge.first)) {
        // Check that the target of the backedge falls inside the current SCS.
        revng_assert(Meta->containsNode(Backedge.second));
        // We need to update the backedges list removing the edges which have
        // been considered as retreatings of the SCS under analysis.
        bool Erased = Backedges.erase(Backedge);
        revng_assert(Erased);
      }
    }

    // A map of candidate entries. The key is a entry candidate, i.e. a node
    // that has an incoming edge from the outer region. The mapped value is the
    // number of edges incoming on the key from an outer region.
    std::map<BasicBlockNodeBB *, size_t> Entries = getCandidateEntries(Meta);
    revng_assert(not Entries.empty());

    // Elect the Entry as the the candidate entry with the largest number of
    // incoming edges from outside the region.
    // If there's a tie, i.e. there are 2 or more candidate entries with the
    // same number of incoming edges from an outer region, we select the entry
    // with the minimal shortest path from entry.
    // It it's still a tie, i.e. there are 2 or more candidate entries with the
    // same number of incoming edges from an outer region and the same minimal
    // shortest path from entry, then we disambiguate by picking the entry that
    // comes first in RPOT.
    BasicBlockNodeBB *Entry = Entries.begin()->first;
    {
      size_t MaxNEntries = Entries.begin()->second;
      size_t ShortestPath = ShortestPathFromEntry.at(Entry);

      auto EntriesEnd = Entries.end();
      for (BasicBlockNodeBB *Node : RPOT) {

        auto EntriesIt = Entries.find(Node);
        if (EntriesIt != EntriesEnd) {

          const auto &[EntryCandidate, NumEntries] = *EntriesIt;
          if (NumEntries > MaxNEntries) {
            Entry = EntryCandidate;
            ShortestPath = ShortestPathFromEntry.at(EntryCandidate);

          } else if (NumEntries == MaxNEntries) {

            size_t SP = ShortestPathFromEntry.at(EntryCandidate);
            if (SP < ShortestPath) {
              Entry = EntryCandidate;
              ShortestPath = SP;
            }
          }
        }
      }
    }

    revng_assert(Entry != nullptr);

    // Print the name of the node that has been selected as head of the region
    revng_log(CombLogger, "Elected head is: " << Entry->getNameStr());

    // Compute the retreating edges and their targets inside the region,
    // starting from the new Entry.
    // Collect the nodes in the metaregion, so we can use the
    // `getBackedgesWhitelist` helper to collect the retreating contained in the
    // current metaregion.
    llvm::SmallSet<BasicBlockNodeBB *, 4> MetaNodes;
    for (BasicBlockNodeBB *Node : Meta->nodes())
      MetaNodes.insert(Node);

    llvm::SmallDenseSet<EdgeDescriptor>
      Retreatings = getBackedgesWhiteList(Entry, MetaNodes).takeSet();
    std::set<BasicBlockNodeBB *> RetreatingTargets;
    for (const EdgeDescriptor &Retreating : Retreatings) {
      revng_log(CombLogger,
                "Retreatings found: " << Retreating.first->getNameStr()
                                      << " -> "
                                      << Retreating.second->getNameStr());

      revng_assert(Meta->containsNode(Retreating.first));
      revng_assert(Meta->containsNode(Retreating.second));
      RetreatingTargets.insert(Retreating.second);
    }

    bool NewHeadNeeded = RetreatingTargets.size() > 1;
    revng_log(CombLogger, "New head needed: " << NewHeadNeeded);

    // Set to contain the retreating edges, which eventually will be connected
    // to the `continue` nodes
    llvm::SmallVector<EdgeDescriptor> ContinueBackedges;
    unsigned DefaultIdx = std::numeric_limits<unsigned>::max();

    BasicBlockNodeBB *Head = Entry;
    if (NewHeadNeeded) {
      // Create the dispatcher.
      Head = RootCFG.addEntryDispatcher();
      Meta->insertNode(Head);

      // For each target of the dispatcher add the edge and add it in the map.
      std::map<std::pair<BasicBlockNodeBB *, std::optional<unsigned>>, unsigned>
        RetreatingIdxMap;
      unsigned IncrementalIdx = 0;

      for (EdgeDescriptor R : Retreatings) {
        BasicBlockNodeBB *OriginalSource = R.first;
        BasicBlockNodeBB *OriginalTarget = R.second;

        // If `OriginalSource` is a `SetNode`, we need to do specific stuff here
        using edge_label_t = typename BasicBlockNodeBB::edge_label_t;
        using EdgeInfo = BasicBlockNodeBB::EdgeInfo;

        std::optional<unsigned> SetIdx = std::nullopt;
        EdgeDescriptor EdgeToRedirect = EdgeDescriptor{ OriginalSource,
                                                        OriginalTarget };

        if (OriginalSource->isSet()) {
          auto *SetUniquePredecessor = OriginalSource->getUniquePredecessor();
          SetIdx = OriginalSource->getStateVariableValue();
          EdgeToRedirect = EdgeDescriptor{ SetUniquePredecessor,
                                           OriginalSource };
        }

        auto MapInsertionIt = RetreatingIdxMap.insert({ { OriginalTarget,
                                                          SetIdx },
                                                        IncrementalIdx });

        // Do different stuff depending if the insertion took place
        unsigned NewIndex = MapInsertionIt.first->second;
        if (bool NewlyInserted = MapInsertionIt.second; NewlyInserted) {
          edge_label_t Labels;
          Labels.insert(NewIndex);
          EdgeInfo EI = { Labels, false };
          addEdge(EdgeDescriptor(Head, EdgeToRedirect.second), EI);
        } else if (OriginalSource->isSet()) {
          // We need to remove the "additional" setnode which will not be used
          RootCFG.removeNode(OriginalSource);
        }
        std::string Name = EdgeToRedirect.second->getName().str();
        auto *SetNode = RootCFG.addEntrySetStateNode(NewIndex, Name);
        Meta->insertNode(SetNode);
        moveEdgeTarget(EdgeToRedirect, SetNode);
        addPlainEdge(EdgeDescriptor(SetNode, Head));
        ContinueBackedges.push_back(EdgeDescriptor(SetNode, Head));
        ++IncrementalIdx;
      }

      // Move the remaining (the retreatings have been handled in the above
      // code) incoming edges from the old head to the new one.
      std::vector<BasicBlockNodeBB *> Predecessors;
      for (BasicBlockNodeBB *Predecessor : Entry->predecessors())
        Predecessors.push_back(Predecessor);

      for (BasicBlockNodeBB *Predecessor : Predecessors) {
        if (not Meta->containsNode(Predecessor)) {
          // We do not expect any SetNode on the edges that are not retreating.
          revng_assert(not Predecessor->isSet());
          moveEdgeTarget(EdgeDescriptor(Predecessor, Entry), Head);
        }
      }

      // We assume that no `SetNode` is present on the default edges.
      DefaultIdx = RetreatingIdxMap.at(std::make_pair(Entry, std::nullopt));
    } else {

      // No head dispatcher has been inserted, so we should insert all the
      // retreating edges in the `ContinueBackedges` set, checking that they
      // point to the `Entry` node
      for (EdgeDescriptor R : Retreatings) {

        // The following should be an assert, but since the backend is in
        // maintenance mode, we have an early return to propagate an early
        // failure
        if (not(R.second == Entry))
          return false;

        ContinueBackedges.push_back(R);
      }
    }

    // Verify that we found at least one backedge
    revng_assert(ContinueBackedges.size() > 0);

    revng_assert(Head != nullptr);
    revng_log(CombLogger, "New head name is: " << Head->getNameStr());

    // Successor refinement step.
    std::set<BasicBlockNodeBB *> Successors = Meta->getSuccessors();
    revng_log(CombLogger, "Initial region successors are:");
    for (BasicBlockNodeBB *Node : Successors) {
      LoggerIndent Indent(CombLogger);
      revng_log(CombLogger, Node->getNameStr());
    }

    revng_log(CombLogger, "Successors Address: " << &Successors);

    bool AnotherIteration = true;
    revng_log(CombLogger, "Adjusting regions successors");
    while (AnotherIteration and Successors.size() > 1) {
      LoggerIndent Indent(CombLogger);
      AnotherIteration = false;
      for (BasicBlockNodeBB *S : llvm::make_early_inc_range(Successors)) {
        revng_log(CombLogger, "Successor: " << S->getID());
        LoggerIndent Indent(CombLogger);

        // If S is already in another metaregion, we don't include it in this
        // one, because that could disrupt the well-nestedness of the meta
        // regions (and possibly force us to recompute the OrderedMetaRegion).
        // TODO: this condition is very likely to be overly strict, because it
        // prevents some good cases to be handled gracefully. In principle I
        // think that we could include any node that is only in the parent
        // region (but not in sibling meta regions), making sure that we never
        // "ingest" a backedge, but this should be thought through before
        // jumping to an implementation.
        if (alreadyInMetaregion(MetaRegions, S)) {
          revng_log(CombLogger, "AlreadyInMetaRegion");
          continue;
        }

        // If any of the predecessors of S is not in the Meta
        // metaregion we don't do anything
        if (llvm::any_of(S->predecessors(), [Meta](auto *P) {
              return not Meta->containsNode(P);
            })) {
          revng_log(CombLogger, "PredecessorIsOutside");
          continue;
        }

        // Otherwise we include S in the Meta metaregion, since
        // all its predecessors are part of it (which means it's
        // dominated by the region).
        revng_assert(not Meta->containsNode(S));
        Meta->insertNode(S);
        revng_log(CombLogger,
                  "Successor has been included in "
                  "the metaregion: "
                    << S->getNameStr());

        // Mark that we want to do another iteration
        AnotherIteration = true;

        // The following is safe because Successors is a std::set
        // and we're using llvm::make_early_inc_range. S has been
        // included in the metaregion, so we have to erase it
        // from Successors, since it's not a successor of the
        // metaregion anymore. Also, all successors of Successor
        // that are not in the metaregion now have to be inserted
        // in Successors, because they are now new successors.
        bool Erased = Successors.erase(S);
        revng_assert(Erased);
        for (BasicBlockNodeBB *NewSuccessor : S->successors()) {
          if (not Meta->containsNode(NewSuccessor)) {
            Successors.insert(NewSuccessor);
            revng_log(CombLogger,
                      "New Successor of the "
                      "metaregion: "
                        << NewSuccessor->getNameStr());
          }
        }
      }

      revng_log(CombLogger, "AnotherIteration: " << AnotherIteration);

      revng_log(CombLogger, "Adjusted region successors are:");
      for (BasicBlockNodeBB *Node : Successors) {
        LoggerIndent Indent(CombLogger);
        revng_log(CombLogger, Node->getNameStr());
      }
    }

    // First Iteration outlining.
    llvm::SmallSet<BasicBlockNodeBB *, 4> OutlinedClonedNodes;
    llvm::SmallSet<BasicBlockNodeBB *, 4> OutlinedOriginalNodes;
    if (Entries.size() > 1) {
      std::map<BasicBlockNodeBB *, BasicBlockNodeBB *> ClonedMap;

      llvm::df_iterator_default_set<BasicBlockNodeBB *> VisitedForOutlining;
      VisitedForOutlining.insert(Head);
      Entries.erase(Head);

      // We perform the cloning of the nodes interested by the first iteration
      // outlining, performing a DFS starting from all the `Entries` nodes, and
      // not proceeding towards node that are not in the `MetaRegion` under
      // restructuring
      for (const auto &[LateEntry, Value] : Entries) {
        auto ItBegin = llvm::df_ext_begin(LateEntry, VisitedForOutlining);
        auto ItEnd = llvm::df_ext_end(LateEntry, VisitedForOutlining);

        while (ItBegin != ItEnd) {

          // Extract the currently visited node
          BasicBlockNodeBB *Node = *ItBegin;

          // If the node is not in the `MetaRegion`, we do not want to proceed
          // in this direction
          if (not Meta->containsNode(Node)) {

            // Skip over the children of `Node`
            ItBegin.skipChildren();

            // Skip the cloning process for the current `Node`, since it is not
            // part of the outlined iteration
            continue;
          }

          // If we reach this point, we are inspecting a node part of the first
          // outlined iteration, therefore we proceed with the cloning
          BasicBlockNodeBB *Clone = RootCFG.cloneNode(*Node);

          // In case we are cloning nodes that may become entry candidates of
          // regions, we need to assign to them a value in the
          // `ShortestPathFromEntry` map
          if (Node->isCollapsed() or Node->isCode()) {
            ShortestPathFromEntry[Clone] = ShortestPathFromEntry.at(Node);
          }
          Clone->setName(Node->getName().str() + " outlined");
          ClonedMap[Node] = Clone;

          // Add the nodes to two additional vectors used later in the
          // postprocessing that assigns each node to the correct `MetaRegion`
          OutlinedClonedNodes.insert(Clone);
          OutlinedOriginalNodes.insert(Node);

          // Increment the `df_iterator`
          ItBegin++;
        }
      }

      // Restore the edges between the node cloned during the first step of the
      // outlining
      for (BasicBlockNodeBB *Node : OutlinedOriginalNodes) {
        revng_assert(Node != Head);

        // Handle the successors of each node
        for (const auto &[Successor, Labels] : Node->labeled_successors()) {
          revng_assert(not Backedges.contains(EdgeDescriptor(Node, Successor)));

          BasicBlockNodeBB *NewEdgeSrc = ClonedMap.at(Node);
          BasicBlockNodeBB *NewEdgeTgt = nullptr;

          if (Meta->containsNode(Successor)) {
            if (OutlinedOriginalNodes.contains(Successor)) {

              // The successor may be another outlined node
              NewEdgeTgt = ClonedMap.at(Successor);
            } else if (Successor == Head) {

              // The successor is the `Head`, so we should reconnect it
              NewEdgeTgt = Head;
            } else {

              // We should not encounter another type of successor
              revng_abort();
            }
          } else {

            // If the successor is not part of the `MetaRegion`, we expect it to
            // be part of the loop successors previously identified
            revng_assert(Successors.contains(Successor));

            NewEdgeTgt = Successor;
          }
          addEdge(EdgeDescriptor(NewEdgeSrc, NewEdgeTgt), Labels);
        }

        // Handle the predecessors. Note that we are interested in handling here
        // only predecessors not belonging to the `MetaRegion`, since the
        // predecessors of each node that lies in the outlined iteration, should
        // have been already handled (or will be) as successors of other nodes
        // in the outlined iteration.
        // We do not iterate directly on the predecessors to avoid iterator
        // invalidation
        llvm::SmallVector<BasicBlockNodeBB *> Predecessors;
        for (BasicBlockNodeBB *Predecessor : Node->predecessors()) {
          Predecessors.push_back(Predecessor);
        }

        for (BasicBlockNodeBB *Predecessor : Predecessors) {
          if (not(Meta->containsNode(Predecessor))) {
            // We handle edges incoming in nodes from outside the outlined
            // iteration

            // Are we moving a backedge with the first iteration outlinig?
            revng_assert(not Backedges.contains({ Predecessor, Node }));

            // If we are on the border of the outlined iteration, `Node` must be
            // one of the late entries
            revng_assert(Entries.contains(Node));

            BasicBlockNodeBB *Clone = ClonedMap.at(Node);
            moveEdgeTarget(EdgeDescriptor(Predecessor, Node), Clone);
          } else {

            // We should do nothing, we already took care of these edges while
            // iterating over the successors of the group of nodes
          }
        }
      }

      // Postprocessing that encapsules each node that has been outlined in the
      // correct `MetaRegion`. The process, at a macro level, proceeds as
      // follows:
      // 1) We need to process all the outlined nodes, with the guarantee that
      //    when we visit each node, all its predecessors must have been already
      //    processed (a requirement for point 3). To do that, we instantiate
      //    multiple `post order` visits, that starts from each successor of
      //    nodes in the outlined iteration. All the visits share the same `ext`
      //    set, which is pre-populated with the nodes that are the entries of
      //    the outlined iteration, so that we only visit nodes we are
      //    interested to postprocess (i.e., the nodes in the outlined
      //    iteration). post order over the `Inverse` graph, using an `ext` DFS
      //    visit that stops at predecessor of the outline iteration.
      // 2) For each node encountered, we proceed at the election of the correct
      //    `MetaRegion` to which the node will be assigned, on the basis of the
      //    following criterion.
      // 3) We collect the predecessors of each node, and collect all the
      //    `MetaRegion`s to which they do belong. After this, we select the
      //    most nested `MetaRegion` between them. This is done under the
      //    assumption that all the candidate `MetaRegion`s must fall on a
      //    single inheritance line on the `MetaRegion` inclusion tree.

      // Collect the `Head` plus the successors, which are the point from where
      // the DFSs for the `po_ext` should start. More in detail, the DFS used by
      // the post order should not start from the `Head` and the `Successors`,
      // but only from their predecessors contained inside the outlined
      // iteration.
      llvm::SmallVector<BasicBlockNodeBB *> BoundaryNodes;
      BoundaryNodes.push_back(Head);
      for (BasicBlockNodeBB *Successor : Successors) {
        BoundaryNodes.push_back(Successor);
      }
      llvm::SmallSet<BasicBlockNodeBB *, 4> DFSOrigins;
      for (BasicBlockNodeBB *BoundaryNode : BoundaryNodes) {
        for (BasicBlockNodeBB *Predecessor : BoundaryNode->predecessors()) {
          if (OutlinedClonedNodes.contains(Predecessor)) {
            DFSOrigins.insert(Predecessor);
            revng_assert(Predecessor != Head);
          }
        }
      }

      // This is the `ext` set used to stop all the subsequents `post order`
      // visits
      llvm::SmallSet<BasicBlockNodeBB *, 4> DFSExtSet;

      // We prepopulate the `DFSExtSet` with all the predecessors of the `Head`
      // that are not part of the outlined iteration
      for (BasicBlockNodeBB *Predecessor : Head->predecessors()) {
        if (not OutlinedClonedNodes.contains(Predecessor)) {
          revng_assert(Predecessor != Head);
          DFSExtSet.insert(Predecessor);
        }
      }

      // We also insert all the predecessors of each `LateEntry`, which are not
      // part of the `OutlinedClonedNodes`
      for (const auto &[LateEntry, Value] : Entries) {
        revng_assert(LateEntry != Head);
        BasicBlockNodeBB *LateEntryCloned = ClonedMap.at(LateEntry);
        for (BasicBlockNodeBB *Predecessor : LateEntryCloned->predecessors()) {
          if (not OutlinedClonedNodes.contains(Predecessor)) {
            revng_assert(Predecessor != Head);
            DFSExtSet.insert(Predecessor);
          }
        }
      }

      // We need to instantiate a new `post order` over the Inverse graph, for
      // each exit point from the outlined iteration. The relevant nodes, have
      // been collected in `DFSOrigins`.
      for (BasicBlockNodeBB *DFSEntry : DFSOrigins) {
        for (BasicBlockNodeBB *OutlinedNode :
             llvm::inverse_post_order_ext(DFSEntry, DFSExtSet)) {

          // We should not encounter a node not part of the outlined iteration
          // during the DFSs
          if (not OutlinedClonedNodes.contains(OutlinedNode)) {
            revng_abort();
          }

          // Find the region where each outlined node should be placed
          MetaRegionBB
            *CandidateParent = computePredecessorsParent(OrderedMetaRegions,
                                                         OutlinedNode);

          // The `CandidateParent` may be `nullptr`, if the `root` region is
          // selected as `CandidateParent`, which is an admissible situation,
          // and since the `root` `MetaRegion` is no more materialized, in such
          // case we do not need to insert the nodes anywhere
          if (CandidateParent != nullptr) {
            CandidateParent->insertNode(OutlinedNode);
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

      llvm::SmallPtrSet<BasicBlockNodeBB *, 8> SetCandidates;
      for (BasicBlockNodeBB *Pred : Head->predecessors())
        if (not Pred->isSet())
          SetCandidates.insert(Pred);

      for (BasicBlockNodeBB *Pred : SetCandidates) {
        BasicBlockNodeBB *Set = RootCFG.addEntrySetStateNode(DefaultIdx,
                                                             Head->getName());
        DefaultEntrySet.push_back(Set);
        EdgeDescriptor PredToHead = { Pred, Head };
        EdgeDescriptor SetToHead = { Set, Head };
        moveEdgeTarget(PredToHead, Set);
        addPlainEdge(SetToHead);

        // Update the backedges set. Basically, when we place the default set
        // node in case of an entry dispatcher, we need to take care to verify
        // if the edge we are "moving" (inserting the set node before it) is a
        // backedge, and in case update the information regarding the backedges
        // present in the graph accordingly (the backedge becomes the edge
        // departing from the set node).
        auto BackEdgeIt = Backedges.find(PredToHead);
        if (BackEdgeIt != Backedges.end()) {
          Backedges.erase(BackEdgeIt);
          Backedges.insert(SetToHead);
        }
      }
    }

    // Exit dispatcher creation.

    // Deduplicate region successor across backedges. If a region has a dummy
    // successor that is a dummy backedge, we want to look across it, so that we
    // can detect if two backedges actually jump to the same target, and emit
    // only one case in the exit dispatcher. This saves us from having to take
    // care later of collapsing the two (or more) dummy branches coming out from
    // the exit dispatcher with different labels. With this strategy we already
    // emit a single label in the first place.
    std::set<BasicBlockNodeBB *> DeduplicatedRegionSuccessors;
    std::map<BasicBlockNodeBB *, BasicBlockNodeBB *> DeduplicationMap;

    // Vector which contains the dummy nodes that are deduplicated during the
    // exit dispatcher creation, and that need to be removed from containing
    // regions.
    std::vector<BasicBlockNodeBB *> DeduplicatedDummies;
    {
      std::map<BasicBlockNodeBB *, BasicBlockNodeBB *> BackedgeToSucc;
      for (BasicBlockNodeBB *Succ : Successors) {
        if (Succ->isEmpty()) {
          revng_assert(Succ->successor_size() == 1);
          BasicBlockNodeBB *BackedgeTgt = *Succ->successors().begin();
          // Lookup if we have already found this backedge target from another
          // exit successor.
          const auto &[It, New] = BackedgeToSucc.insert({ BackedgeTgt, Succ });
          if (New) {
            // If we haven't, add the successor in the deduplicated successors
            DeduplicatedRegionSuccessors.insert(Succ);
            DeduplicationMap[Succ] = Succ;
          } else {
            // If we have, map the successor to the old successor we've found
            // with the same backedge target.
            DeduplicationMap[Succ] = It->second;

            // We can have two situations:
            // 1: The dummy whose use we have simplified, was connected only
            //    through the path we have simplified away, which coming from
            //    the inside of the `MetaRegion`. Therefore we can proceed in
            //    actually removing such dummy.
            // 2: Another path not coming from nodes inside the `MetaRegion`.
            //    This can happen if, e.g., first iteration outlining cloned the
            //    path reaching the dummy. If we simplify the dummy, we perform
            //    an incorrect operation.
            if (llvm::all_of(Succ->predecessors(), [Meta](auto *P) {
                  return Meta->containsNode(P);
                })) {

              // If we are following this way of collapsing the successors
              // edges, it means that we are collapsing two different retreating
              // edges on a single retreating, so a backedge entry will remain
              // in the global `Backedges` set as a ghost entry, and we need to
              // take care of removing it now.
              Backedges.erase({ Succ, BackedgeTgt });

              // If we are "using" another `Dummy` node for representing the
              // backedge, we need to take into consideration that the current
              // dummy will need to be recursively removed from parent regions
              // that contain the current region, otherwise a "ghost" node will
              // remain in them.
              DeduplicatedDummies.push_back(Succ);
            }
          }
        } else {
          DeduplicatedRegionSuccessors.insert(Succ);
          DeduplicationMap[Succ] = Succ;
        }
      }
    }

    bool NewExitNeeded = DeduplicatedRegionSuccessors.size() > 1;
    revng_log(CombLogger, "New exit needed: " << NewExitNeeded);

    BasicBlockNodeBB *ExitDispatcher = nullptr;
    if (NewExitNeeded) {

      // Create the dispatcher.
      ExitDispatcher = RootCFG.addExitDispatcher();

      // For each target of the dispatcher add the edge and add it in the map.
      std::map<BasicBlockNodeBB *, unsigned> SuccessorsIdxMap;
      for (auto &Group : llvm::enumerate(DeduplicatedRegionSuccessors)) {
        BasicBlockNodeBB *Successor = Group.value();
        unsigned Idx = Group.index();

        SuccessorsIdxMap[Successor] = Idx;

        using edge_label_t = typename BasicBlockNodeBB::edge_label_t;
        edge_label_t Labels;
        Labels.insert(Idx);
        using EdgeInfo = typename BasicBlockNodeBB::EdgeInfo;
        EdgeInfo EI = { Labels, false };
        addEdge(EdgeDescriptor(ExitDispatcher, Successor), EI);
      }

      std::set<EdgeDescriptor> OutEdges = Meta->getOutEdges();
      for (EdgeDescriptor Edge : OutEdges) {
        // We should not be adding new backedges.
        revng_assert(not Backedges.contains(Edge));

        unsigned Idx = SuccessorsIdxMap.at(DeduplicationMap.at(Edge.second));
        auto *IdxSetNode = RootCFG.addExitSetStateNode(Idx,
                                                       Edge.second->getName());
        Meta->insertNode(IdxSetNode);
        moveEdgeTarget(Edge, IdxSetNode);
        addPlainEdge(EdgeDescriptor(IdxSetNode, Edge.second));
      }

      revng_log(CombLogger,
                "New exit name is: " << ExitDispatcher->getNameStr());
    }

    // Collapse Region.
    // Create a new RegionCFG object for representing the collapsed region and
    // populate it with the internal nodes.
    Regions.push_back(RegionCFG<BasicBlock *>());
    RegionCFG<BasicBlock *> &CollapsedGraph = Regions.back();
    RegionCFG<BasicBlock *>::BBNodeMap SubstitutionMap{};
    CollapsedGraph.setFunctionName(F.getName().str());
    CollapsedGraph.setRegionName(std::to_string(Meta->getIndex()));
    revng_assert(Head != nullptr);

    // Create the collapsed node in the outer region.
    BasicBlockNodeBB *Collapsed = RootCFG.createCollapsedNode(&CollapsedGraph);

    // A collapsed node may become a candidate entry for an outer cyclic region
    // so we need to assign to it a value in the `ShortestPathFromEntry` map.
    ShortestPathFromEntry[Collapsed] = ShortestPathFromEntry[Head];

    {
      // Update the backedges set, checking that if a backedge of an outer
      // region pointed to a node that now has been collapsed, now should point
      // to the collapsed node, and that does not exists at this point a
      // backedge which has as source a node that will be collapsed.
      // We cannot do it in a single iteration because `llvm::SmallDenseSet`
      // invalidates the iterators upon an erase and insertion operation.
      bool Changed = true;
      while (Changed) {
        Changed = false;
        for (const auto &Backedge : Backedges) {
          auto &&[Source, Target] = Backedge;
          revng_assert(not Meta->containsNode(Source));
          if (Meta->containsNode(Target)) {
            revng_assert(Target == Head);
            Backedges.erase(Backedge);
            Backedges.insert({ Source, Collapsed });
            Changed = true;
            break;
          }
        }
      }
    }

    // Creation and connection of the break and continue node is now performed
    // during the bulk node insertion, in order to avoid errors in edge
    // ordering.
    std::set<EdgeDescriptor> OutgoingEdges = Meta->getOutEdges();

    // The following call may fail, and in that case we propagate the error
    // upwards
    if (not CollapsedGraph.insertBulkNodes(Meta->getNodes(),
                                           Head,
                                           SubstitutionMap,
                                           OutgoingEdges,
                                           ContinueBackedges)) {
      return false;
    }

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
      revng_assert(ExitDispatcher != nullptr);
      addPlainEdge(EdgeDescriptor(Collapsed, ExitDispatcher));
    } else {

      // Double check that we have at most a single successor
      revng_assert(DeduplicatedRegionSuccessors.size() <= 1);
      if (DeduplicatedRegionSuccessors.size() == 1) {

        // Connect the collapsed node to the unique successor
        BasicBlockNodeBB *Successor = *DeduplicatedRegionSuccessors.begin();
        addPlainEdge(EdgeDescriptor(Collapsed, Successor));
      }
    }

    // Remove collapsed nodes from the outer region.
    for (BasicBlockNodeBB *Node : Meta->nodes()) {
      revng_log(CombLogger,
                "Removing from main graph node :" << Node->getNameStr());
      RootCFG.removeNode(Node);
      llvm::erase_value(RPOT, Node);
    }

    LogMetaRegions(OrderedMetaRegions, "MetaRegions before update");
    // Substitute in the other SCSs the nodes of the current SCS with the
    // collapsed node and the exit dispatcher structure.
    MetaRegionBB *ParentMetaRegion = Meta->getParent();
    while (ParentMetaRegion) {
      ParentMetaRegion->updateNodes(Meta->getNodes(),
                                    Collapsed,
                                    ExitDispatcher,
                                    DefaultEntrySet,
                                    DeduplicatedDummies);
      ParentMetaRegion = ParentMetaRegion->getParent();
    }
    LogMetaRegions(OrderedMetaRegions, "MetaRegions after update");

    // Replace the pointers inside SCS.
    Meta->replaceNodes(CollapsedGraph.getNodes());

    // Remove useless nodes inside the SCS (like dandling break/continue)
    CollapsedGraph.removeNotReachables(OrderedMetaRegions);

    // Serialize the newly collapsed SCS region.
    if (CombLogger.isEnabled()) {
      CombLogger << "Dumping CFG of metaregion " << Meta->getIndex() << "\n";
      CollapsedGraph.dumpCFGOnFile(F.getName().str(),
                                   "restructure",
                                   "region-" + std::to_string(Meta->getIndex())
                                     + "-inside");
      CombLogger << "Dumping main graph snapshot post restructuring\n";
      RootCFG.dumpCFGOnFile(F.getName().str(),
                            "restructure",
                            "region-" + std::to_string(Meta->getIndex())
                              + "-outside-after");
    }

    // Remove not reachables nodes from the graph at each iteration.
    RootCFG.removeNotReachables(OrderedMetaRegions);

    // Check that the newly created collapsed region is acyclic.
    revng_assert(CollapsedGraph.isDAG());
  }

  // After the restructuring of all the metaregions, we need to ensure that all
  // the backedges contained in the `Backedges` global set have been taken care
  // of.
  revng_assert(Backedges.empty());

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph before final purge\n";
    RootCFG.dumpCFGOnFile(F.getName().str(),
                          "restructure",
                          "final-state-before-purge");
  }

  // Remove not reachables nodes from the main final graph.
  RootCFG.removeNotReachables(OrderedMetaRegions);

  // Serialize the newly collapsed SCS region.
  if (CombLogger.isEnabled()) {
    CombLogger << "Dumping main graph after final purge\n";
    RootCFG.dumpCFGOnFile(F.getName().str(),
                          "restructure",
                          "final-state-after-purge");
  }

  // Print metaregions after ordering.
  LogMetaRegions(OrderedMetaRegions, "Metaregions after collapse:");

  // Check that the root region is acyclic at this point.
  revng_assert(RootCFG.isDAG());

  // Collect statistics
  unsigned InitialWeight = 0;
  if (MetricsOutputPath.getNumOccurrences()) {
    revng_assert(MetricsOutputPath.getNumOccurrences() == 1);
    // Compute the initial weight of the CFG.
    for (BasicBlockNodeBB *BBNode : RootCFG.nodes()) {
      InitialWeight += BBNode->getWeight();
    }
  }

  // Invoke the AST generation for the root region.
  std::map<RegionCFG<llvm::BasicBlock *> *, ASTTree> CollapsedMap;
  if (not generateAst(RootCFG, AST, CollapsedMap))

    // We propagate the failure upwards
    return false;

  // Scorporated this part which was previously inside the `generateAst` to
  // avoid having it run twice or more (it was run inside the recursive step
  // of the `generateAst`, and then another time for the final root AST, which
  // now is directly the entire AST, since there's no flattening anymore).
  normalize(AST, F);

  // Serialize the collected metrics in the outputfile.
  if (MetricsOutputPath.getNumOccurrences()) {
    // Compute the increase in weight, on the AST
    unsigned FinalWeight = 0;
    for (ASTNode *N : AST.nodes()) {
      switch (N->getKind()) {
      case ASTNode::NK_Scs:
      case ASTNode::NK_If:
      case ASTNode::NK_Switch: {
        // Control-flow nodes emit single constructs, so we just increase the
        // weight by one.
        // Control-flow nodes would also have nested scopes (then-else for if,
        // cases for switch, loop body for scs). However, those nodes are
        // visited separately, and will be accounted for later.
        ++FinalWeight;
      } break;
      case ASTNode::NK_Set:
      case ASTNode::NK_Break:
      case ASTNode::NK_SwitchBreak:
      case ASTNode::NK_Continue: {
        // These AST Nodes are emitted as single instructions.
        // Just increase the weight by one.
        ++FinalWeight;
      } break;
      case ASTNode::NK_List: {
        // Sequence nodes are just scopes, they don't have a real weight.
        // Their weight is just sum of the weights of the nodes they contain,
        // that will be visited nevertheless.
      } break;
      case ASTNode::NK_Code: {
        auto *BB = cast<CodeNode>(N)->getOriginalBB();
        revng_assert(BB);
        FinalWeight += WeightTraits<llvm::BasicBlock *>::getWeight(BB);
      } break;
      default:
        revng_abort("unexpected AST node");
      }
    }

    float Increase = float(FinalWeight) / float(InitialWeight);

    std::ofstream Output;
    const char *FunctionName = F.getName().data();
    std::ostream &OutputStream = pathToStream(MetricsOutputPath + "/"
                                                + FunctionName,
                                              Output);
    OutputStream << "function,"
                    "duplications,percentage,tuntangle,puntangle,iweight\n";
    OutputStream << F.getName().data() << "," << DuplicationCounter << ","
                 << Increase << "," << UntangleTentativeCounter << ","
                 << UntanglePerformedCounter << "," << InitialWeight << "\n";
  }

  // We return true to notify that not restructuring error arose
  return true;
}
