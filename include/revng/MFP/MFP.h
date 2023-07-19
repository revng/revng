#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <cstddef>
#include <map>
#include <queue>
#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/ReversePostOrderTraversal.h"

namespace MFP {

template<typename LatticeElement>
struct MFPResult {
  LatticeElement InValue;
  LatticeElement OutValue;
};

/// GT is an instance of llvm::GraphTraits e.g. llvm::GraphTraits<GraphType>
template<typename GT>
auto successors(typename GT::NodeRef From) {
  return llvm::make_range(GT::child_begin(From), GT::child_end(From));
}

template<typename MFI, typename LatticeElement = typename MFI::LatticeElement>
concept MonotoneFrameworkInstance = requires(const MFI &I,
                                             LatticeElement E1,
                                             LatticeElement E2,
                                             typename MFI::Label L) {
  /// To compute the reverse post order traversal of the graph starting from
  /// the extremal nodes, we need that the nodes also represent a subgraph
  typename llvm::GraphTraits<typename MFI::Label>::NodeRef;

  std::same_as<typename MFI::Label,
               typename llvm::GraphTraits<typename MFI::GraphType>::NodeRef>;
  { I.combineValues(E1, E2) } -> std::same_as<LatticeElement>;
  { I.isLessOrEqual(E1, E2) } -> std::same_as<bool>;
  { I.applyTransferFunction(L, E2) } -> std::same_as<LatticeElement>;
};

template<typename Label, typename LatticeElement>
using ResultMap = std::map<Label, MFPResult<LatticeElement>>;

template<MonotoneFrameworkInstance MFI>
using MFIResultMap = ResultMap<typename MFI::Label,
                               typename MFI::LatticeElement>;

/// Compute the maximum fixed points of an instance of monotone framework GT an
/// instance of llvm::GraphTraits that tells us how to visit the graph LGT a
/// graph type that tells us how to visit the subgraph induced by a node in the
/// graph. This is needed for the RPOT because for certain graph (e.g.
/// Inverse<...>) the nodes don't necessary carry all the information that
/// GraphType has.
template<MonotoneFrameworkInstance MFI,
         typename GT = llvm::GraphTraits<typename MFI::GraphType>,
         typename LGT = typename MFI::Label>
MFIResultMap<MFI>
getMaximalFixedPoint(const MFI &Instance,
                     typename MFI::GraphType Flow,
                     typename MFI::LatticeElement InitialValue,
                     typename MFI::LatticeElement ExtremalValue,
                     const std::vector<typename MFI::Label> &ExtremalLabels,
                     const std::vector<typename MFI::Label> &InitialNodes) {
  using Label = typename MFI::Label;
  using LatticeElement = typename MFI::LatticeElement;

  std::map<Label, LatticeElement> PartialAnalysis;
  std::map<Label, MFPResult<LatticeElement>> AnalysisResult;

  struct WorklistItem {
    size_t Priority;
    Label Item;

    std::strong_ordering operator<=>(const WorklistItem &) const = default;
  };
  std::set<WorklistItem> Worklist;

  llvm::SmallSet<Label, 8> Visited{};
  std::map<Label, size_t> LabelPriority;

  // Step 1 initialize the worklist and extremal labels
  for (Label ExtremalLabel : ExtremalLabels) {
    AnalysisResult[ExtremalLabel].InValue = ExtremalValue;
  }

  for (Label Start : InitialNodes) {
    if (!Visited.contains(Start)) {
      // Fill the worklist with nodes in reverse post order
      // launching a visit from each remaining node
      ReversePostOrderTraversalExt<LGT,
                                   llvm::GraphTraits<LGT>,
                                   llvm::SmallSet<Label, 8>>
        RPOTE(Start, Visited);
      for (Label Node : RPOTE) {
        LabelPriority[Node] = LabelPriority.size();
        Worklist.insert({ LabelPriority.at(Node), Node });
        // initialize the analysis value for non extremal nodes
        if (!AnalysisResult.contains(Node))
          AnalysisResult[Node].InValue = InitialValue;
      }
    }
  }

  // Step 2 iteration
  while (!Worklist.empty()) {
    WorklistItem First = *Worklist.begin();
    Label Start = First.Item;
    Worklist.erase(First);
    auto &LabelAnalysis = AnalysisResult.at(Start);
    LabelAnalysis
      .OutValue = Instance.applyTransferFunction(Start, LabelAnalysis.InValue);

    for (Label End : successors<GT>(Start)) {
      auto &PartialEnd = AnalysisResult.at(End);
      if (!Instance.isLessOrEqual(LabelAnalysis.OutValue, PartialEnd.InValue)) {
        PartialEnd.InValue = Instance.combineValues(PartialEnd.InValue,
                                                    LabelAnalysis.OutValue);
        Worklist.insert({ LabelPriority.at(End), End });
      }
    }
  }

  return AnalysisResult;
}

template<MonotoneFrameworkInstance MFI,
         typename GT = llvm::GraphTraits<typename MFI::GraphType>,
         typename LGT = typename MFI::Label>
MFIResultMap<MFI>
getMaximalFixedPoint(const MFI &Instance,
                     typename MFI::GraphType Flow,
                     typename MFI::LatticeElement InitialValue,
                     typename MFI::LatticeElement ExtremalValue,
                     const std::vector<typename MFI::Label> &ExtremalLabels) {
  using Label = typename MFI::Label;
  std::vector<Label> InitialNodes(ExtremalLabels);

  // Handle the special case that the graph has a single entry node
  if (GT::getEntryNode(Flow) != nullptr) {
    InitialNodes.push_back(GT::getEntryNode(Flow));
  }
  // Start visits for nodes that we still haven't visited
  // prioritizing extremal nodes
  for (Label Node :
       llvm::make_range(GT::nodes_begin(Flow), GT::nodes_end(Flow))) {
    InitialNodes.push_back(Node);
  }
  return getMaximalFixedPoint<MFI, GT, LGT>(Instance,
                                            Flow,
                                            InitialValue,
                                            ExtremalValue,
                                            ExtremalLabels,
                                            InitialNodes);
}

} // namespace MFP
