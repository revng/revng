//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <algorithm>
#include <functional>
#include <limits>
#include <random>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

#include "ContractedGraph.h"
#include "Mincut.h"
#include "TypeFlowNode.h"

using namespace vma;

static Logger<> MincutLog("vma-mincut");

static llvm::cl::opt<unsigned> MincutIterOpt("vma-mincut-iter",
                                             llvm ::cl::desc("Specify the "
                                                             "number of karger "
                                                             "iterations for "
                                                             "the mincut "
                                                             "algorithm"));

static unsigned calcCost(ContractedGraph &G) {
  auto ContractedSize = G.NodesToColor->totalSize()
                        + G.NodesToUncolor->totalSize();
  revng_assert(G.NTypeFlowNodes == ContractedSize);
  unsigned Cost = 0;

  llvm::SmallSet<TypeFlowNode *, 16> Visited;

  // Cost of a node in the NodesToColor set
  auto CostOfNodeToColor = [&Visited, &G](TypeFlowNode *TFGNode) {
    // Pay the cost only for the nodes that are being decided by the mincut
    if (not TFGNode->isUndecided())
      return 0U;

    unsigned AdditionalCost = 0;

    // If the node is undecided and belongs to NodesToColor, it means that all
    // of its successors that have the right color are also in NodesToColor.
    // This means that successors that do not belong to NodesToColor have
    // automatically the wrong color.
    for (auto *Succ : TFGNode->successors())
      if (not G.NodesToColor->contains(Succ))
        AdditionalCost++;

    Visited.insert(TFGNode);
    return AdditionalCost;
  };

  for (auto *TFGNode : G.NodesToColor->InitialNodes)
    Cost += CostOfNodeToColor(TFGNode);
  for (auto *TFGNode : G.NodesToColor->AdditionalNodes)
    Cost += CostOfNodeToColor(TFGNode);

  // Cost of a node in the NodesToUncolor set
  auto CostOfNodeToUncolor = [&Visited, &G](TypeFlowNode *TFGNode) {
    // Pay the cost only for the nodes that are being decided by the mincut
    if (not TFGNode->isUndecided())
      return 0U;

    unsigned AdditionalCost = 0;

    // If the node belongs to NodesToUncolor, remove G.Color from the candidates
    ColorSet NodeColor = TFGNode->getCandidates();
    NodeColor.Bits.reset(G.Color.firstSetBit());

    for (auto *Succ : TFGNode->successors()) {
      if (Visited.contains(Succ))
        continue;

      ColorSet CommonColors;
      CommonColors.Bits = Succ->getCandidates().Bits & NodeColor.Bits;
      // If the node and its successor have no common candidates, pay a cost
      if (CommonColors.countValid() == 0)
        AdditionalCost++;
    }

    Visited.insert(TFGNode);
    return AdditionalCost;
  };

  for (auto *TFGNode : G.NodesToUncolor->InitialNodes)
    Cost += CostOfNodeToUncolor(TFGNode);
  for (auto *TFGNode : G.NodesToUncolor->AdditionalNodes)
    Cost += CostOfNodeToUncolor(TFGNode);

  return Cost;
}

void vma::karger(ContractedGraph &G,
                 unsigned &BestCost,
                 ContractedNode &BestNodesToColor,
                 ContractedNode &BestNodesToUncolor) {

  // Fixed seed generated with /dev/urandom
  static const unsigned RandSeed = 320464148;
  // Seed the random generator for repeatability
  srand(RandSeed);

  // TODO: find a sane default, e.g. 10 * log2 (G.size())
  const unsigned int DefaultNIter = 50U;
  const unsigned NIter = (MincutIterOpt ? MincutIterOpt : DefaultNIter);

  // Execute many times (Monte-carlo)
  for (size_t Iter = 0; Iter < NIter; Iter++) {
    G.reset();

    auto SpecialNodesDimension = [&G]() {
      return G.NodesToColor->totalSize() + G.NodesToUncolor->totalSize();
    };

    // Execute Karger until all nodes have been collapsed in a special supernode
    while (G.NTypeFlowNodes > SpecialNodesDimension()) {
      unsigned RandIdx = rand() % G.NActiveEdges;
      G.contract(RandIdx);
    }

    if (VerifyLog.isEnabled())
      G.check();

    unsigned Cost = calcCost(G);

    // Update best solution
    if (Cost < BestCost) {
      BestCost = Cost;
      std::swap(BestNodesToColor.AdditionalNodes,
                G.NodesToColor->AdditionalNodes);
      std::swap(BestNodesToUncolor.AdditionalNodes,
                G.NodesToUncolor->AdditionalNodes);

      revng_log(MincutLog,
                "Karger new best cost: " << BestCost << " [iteration: " << Iter
                                         << "]");

      revng_log(MincutLog, "Best choice: divided");
    }

    if (BestCost == 0)
      break;
  }
}

/// Generate the solution in which all nodes of \a G are colored
static void generateColorAllSolution(ContractedGraph &G) {
  for (auto &CN : G.Nodes) {
    if (CN.get() == G.NodesToColor or CN.get() == G.NodesToUncolor)
      continue;

    for (TypeFlowNode *TFGNode : CN->InitialNodes) {
      revng_assert(not TFGNode->isDecided()
                   or not TFGNode->getCandidates().contains(G.Color));
      G.NodesToColor->AdditionalNodes.insert(TFGNode);
      G.getMapEntry(TFGNode) = G.NodesToColor;
    }
  }
}

/// Generate the solution in which all nodes of \a G are uncolored
static void moveAllColoredToUncolored(ContractedGraph &G) {
  std::swap(G.NodesToUncolor->AdditionalNodes, G.NodesToColor->AdditionalNodes);
  for (TypeFlowNode *TFGNode : G.NodesToUncolor->AdditionalNodes) {
    revng_assert(not TFGNode->isDecided()
                 or not TFGNode->getCandidates().contains(G.Color));
    G.getMapEntry(TFGNode) = G.NodesToUncolor;
  }
}

/// Generate the two simplest cuts (color all and uncolor all)
static void generateNaiveSolutions(ContractedGraph &G,
                                   unsigned &BestCost,
                                   ContractedNode &BestNodesToColor,
                                   ContractedNode &BestNodesToUncolor) {
  generateColorAllSolution(G);
  unsigned ColorAllCost = calcCost(G);

  moveAllColoredToUncolored(G);
  unsigned RemoveAllCost = calcCost(G);

  revng_log(MincutLog,
            "cost of coloring all: "
              << ColorAllCost << "  cost of removing all: " << RemoveAllCost);

  if (RemoveAllCost < ColorAllCost) {
    BestCost = RemoveAllCost;
    BestNodesToColor.AdditionalNodes.clear();
    std::swap(BestNodesToUncolor.AdditionalNodes,
              G.NodesToUncolor->AdditionalNodes);

    revng_log(MincutLog, "Best choice: remove all");
  } else {
    BestCost = ColorAllCost;
    std::swap(BestNodesToColor.AdditionalNodes,
              G.NodesToUncolor->AdditionalNodes);
    BestNodesToUncolor.AdditionalNodes.clear();

    revng_log(MincutLog, "Best choice: color all");
  }
}

void vma::minCut(TypeFlowGraph &TG) {
  // Apply karger one color at a time, using the color index in the bitset as
  // ordering criterion.
  for (unsigned I = 0; I < MAX_COLORS; I++) {
    ColorSet CurColor(1 << I);

    revng_log(MincutLog, "------ Color: " << dumpToString(CurColor));

    for (TypeFlowNode *N : TG.nodes()) {
      // Check if we can start building a ContractedGraph from the current node
      auto HasUndecidedNeighbors = [CurColor](TypeFlowNode *TFGNodeode) {
        return llvm::any_of(TFGNodeode->successors(),
                            [CurColor](TypeFlowNode *Succ) {
                              auto SuccCandidates = Succ->getCandidates();
                              return Succ->isUndecided()
                                     and SuccCandidates.contains(CurColor);
                            });
      };
      if (not(N->isDecided() and N->getCandidates().contains(CurColor)
              and HasUndecidedNeighbors(N)))
        continue;

      // Build Contracted graph
      ContractedGraph G{ CurColor };
      makeContractedGraph(G, N, CurColor);

      if (VerifyLog.isEnabled())
        G.check();

      revng_log(MincutLog, "------ New karger: " << G.NTypeFlowNodes);
      revng_log(MincutLog,
                "Karger with "
                  << G.NTypeFlowNodes
                  << " nodes, NodesToColor: " << G.NodesToColor->totalSize()
                  << "  NodesToUncolor: " << G.NodesToUncolor->totalSize());

      // Keep track of the best solution
      unsigned BestCost = std::numeric_limits<unsigned>::max();
      ContractedNode BestNodesToColor = *G.NodesToColor;
      ContractedNode BestNodesToUncolor = *G.NodesToUncolor;

      // Try to color all and uncolor all
      generateNaiveSolutions(G, BestCost, BestNodesToColor, BestNodesToUncolor);

      // If NodesToUncolor is empty there's no point in trying karger
      if (G.NodesToUncolor->InitialNodes.size() > 0)
        karger(G, BestCost, BestNodesToColor, BestNodesToUncolor);

      revng_log(MincutLog,
                "Final solution "
                  << G.NTypeFlowNodes
                  << " nodes, NodesToColor: " << BestNodesToColor.totalSize()
                  << "  NodesToUncolor: " << BestNodesToUncolor.totalSize());

      // Color all nodes that belong to NodesToColor
      for (TypeFlowNode *TFGNode : BestNodesToColor.InitialNodes) {
        revng_assert(TFGNode->getCandidates().contains(G.Color));
        TFGNode->setCandidates(G.Color);
      }
      for (TypeFlowNode *TFGNode : BestNodesToColor.AdditionalNodes) {
        revng_assert(TFGNode->getCandidates().contains(G.Color));
        TFGNode->setCandidates(G.Color);
      }
      // Uncolor all nodes that belong to NodesToColor
      for (TypeFlowNode *TFGNode : BestNodesToUncolor.InitialNodes) {
        auto InitialColor = TFGNode->getCandidates();

        revng_assert(not TFGNode->isDecided()
                     or not InitialColor.contains(G.Color));

        InitialColor.Bits.reset(I);
        TFGNode->setCandidates(InitialColor);
      }
      for (TypeFlowNode *TFGNode : BestNodesToUncolor.AdditionalNodes) {
        auto InitialColor = TFGNode->getCandidates();

        revng_assert(not TFGNode->isDecided()
                     or not InitialColor.contains(G.Color));
        InitialColor.Bits.reset(I);
        TFGNode->setCandidates(InitialColor);
      }

      revng_log(MincutLog, "CurCost after applying mincut  " << countCasts(TG));
    }

    // Remove CurColor from the candidates of any remaining grey node before
    // going to another color
    for (TypeFlowNode *N : TG.nodes()) {
      auto InitialColor = N->getCandidates();

      if (N->isUndecided() and InitialColor.contains(CurColor)) {
        InitialColor.Bits.reset(I);
        N->setCandidates(InitialColor);
      }
    }

    revng_log(MincutLog, "CurCost after resetting color  " << countCasts(TG));

    applyMajorityVoting(TG);
    revng_log(MincutLog,
              "CurCost after applying majority voting  " << countCasts(TG));
  }
}
