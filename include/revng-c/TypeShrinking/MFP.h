#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <cstddef>
#include <map>
#include <queue>
#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/GenericGraph.h"

#include "revng-c/ADT/ReversePostOrderTraversal.h"

namespace TypeShrinking {

template<typename T, typename U>
concept same_as = std::is_same_v<T, U>;

template<typename LatticeElement>
struct MFPResult {
  LatticeElement inValue;
  LatticeElement outValue;
};

/// GT is an instance of llvm::GraphTraits e.g. llvm::GraphTraits<GraphType>
template<typename GT>
auto successors(typename GT::NodeRef From) {
  return llvm::make_range(GT::child_begin(From), GT::child_end(From));
}

template<typename MFI>
concept MonotoneFrameworkInstance = requires(typename MFI::LatticeElement E1,
                                             typename MFI::LatticeElement E2,
                                             typename MFI::Label L) {
  /// To compute the reverse post order traversal of the graph starting from
  /// the extremal nodes, we need that the nodes also represent a subgraph
  typename llvm::GraphTraits<typename MFI::Label>::NodeRef;

  same_as<typename MFI::Label,
          typename llvm::GraphTraits<typename MFI::GraphType>::NodeRef>;
  // Disable clang-format, because it does not handle concepts very well yet
  // clang-format off
  { MFI::combineValues(E1, E2) } ->same_as<typename MFI::LatticeElement>;
  { MFI::isLessOrEqual(E1, E2) } ->same_as<bool>;
  { MFI::applyTransferFunction(L, E2) } ->same_as<typename MFI::LatticeElement>;
  // clang-format on
};

template<typename Label>
struct WorklistItem {
  size_t priority;
  Label label;
  friend bool
  operator<(const WorklistItem<Label> &a, const WorklistItem<Label> &b) {
    return a.priority < b.priority;
  }
  friend bool
  operator==(const WorklistItem<Label> &a, const WorklistItem<Label> &b) {
    return a.label < b.label;
  }
};

/// Compute the maximum fixed points of an instance of monotone framework
/// GT an instance of llvm::GraphTraits
template<MonotoneFrameworkInstance MFI,
         typename GT = llvm::GraphTraits<typename MFI::GraphType>>
std::map<typename MFI::Label, MFPResult<typename MFI::LatticeElement>>
getMaximalFixedPoint(const typename MFI::GraphType &Flow,
                     typename MFI::LatticeElement InitialValue,
                     typename MFI::LatticeElement ExtremalValue,
                     const std::vector<typename MFI::Label> &ExtremalLabels) {
  typedef typename MFI::Label Label;
  typedef typename MFI::LatticeElement LatticeElement;
  std::map<Label, LatticeElement> PartialAnalysis;
  std::map<Label, MFPResult<LatticeElement>> AnalysisResult;
  std::set<WorklistItem<Label>> Worklist;

  // Step 1 initialize the worklist and extremal labels
  for (Label ExtremalLabel : ExtremalLabels) {
    PartialAnalysis[ExtremalLabel] = ExtremalValue;
  }

  llvm::SmallSet<Label, 8> Visited{};
  std::map<Label, size_t> LabelPriority;
  for (Label Start : llvm::nodes(Flow)) {
    if (Visited.count(Start) == 0) {
      // fill the worklist with nodes in reverse post order
      // lauching a visit from each remaining node
      ReversePostOrderTraversalExt RPOTE(Start, Visited);
      for (Label Node : RPOTE) {
        LabelPriority[Node] = LabelPriority.size();
        Worklist.insert({ LabelPriority.at(Node), Node });
        // initialize the analysis value for non extremal nodes
        if (PartialAnalysis.find(Node) == PartialAnalysis.end()) {
          PartialAnalysis[Node] = InitialValue;
        }
      }
    }
  }

  // Step 2 iteration
  while (!Worklist.empty()) {
    WorklistItem<Label> First = *Worklist.begin();
    Label Start = First.label;
    Worklist.erase(First);
    for (Label End : successors<GT>(Start)) {
      auto &PartialStart = PartialAnalysis.at(Start);
      LatticeElement
        UpdatedEndAnalysis = MFI::applyTransferFunction(Start, PartialStart);
      auto &PartialEnd = PartialAnalysis.at(End);
      if (!MFI::isLessOrEqual(UpdatedEndAnalysis, PartialEnd)) {
        PartialEnd = MFI::combineValues(PartialEnd, UpdatedEndAnalysis);
        Worklist.insert({ LabelPriority.at(End), End });
      }
    }
  }

  // Step 3 presenting the results
  for (auto &[Node, Analysis] : PartialAnalysis) {
    AnalysisResult[Node] = { Analysis,
                             MFI::applyTransferFunction(Node, Analysis) };
  }
  return AnalysisResult;
}

} // namespace TypeShrinking
