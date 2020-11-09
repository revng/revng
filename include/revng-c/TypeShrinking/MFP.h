#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <functional>
#include <map>
#include <queue>
#include <type_traits>
#include <unordered_map>

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

template<typename LatticeElement,
         typename GraphType,
         class MonotoneFrameworkInstance>
struct MonotoneFramework {
  // by the specs of llvm::GraphTraits, NodeRef chould be cheap to copy
  using GT = llvm::GraphTraits<GraphType>;
  using Label = typename GT::NodeRef;

  using LatticeElementPair = std::pair<LatticeElement, LatticeElement>;

  static LatticeElement
  combineValues(const LatticeElement &lh, const LatticeElement &rh) {
    return MonotoneFrameworkInstance::combineValues(lh, rh);
  }

  static LatticeElement
  applyTransferFunction(Label L, const LatticeElement &E) {
    return MonotoneFrameworkInstance::applyTransferFunction(L, E);
  }

  static bool
  isLessOrEqual(const LatticeElement &lh, const LatticeElement &rh) {
    return MonotoneFrameworkInstance::isLessOrEqual(lh, rh);
  }

  /// Compute the maximum fixed points of an instance of monotone framework
  std::map<Label, LatticeElementPair>
  getMaximalFixedPoint(const GraphType &Flow,
                       LatticeElement BottomValue,
                       LatticeElement ExtremalValue,
                       const std::vector<Label> &ExtremalLabels) {
    std::map<Label, LatticeElement> PartialAnalysis;
    std::map<Label, LatticeElementPair> AnalysisResult;
    std::queue<std::pair<Label, Label>> Worklist;

    // Step 1.1 initialize the worklist and extremal labels
    for (Label Start : llvm::nodes(Flow)) {
      PartialAnalysis[Start] = BottomValue;

      for (auto End : successors(Start)) {
        Worklist.push({ Start, End });
      }
    }
    for (auto ExtremalLabel : ExtremalLabels) {
      PartialAnalysis[ExtremalLabel] = ExtremalValue;
    }

    // Step 2 iteration
    while (!Worklist.empty()) {
      auto [Start, End] = Worklist.front();
      Worklist.pop();
      auto &Partial = PartialAnalysis[Start];
      auto UpdatedEndAnalysis = applyTransferFunction(Start, Partial);
      if (!isLessOrEqual(UpdatedEndAnalysis, PartialAnalysis[End])) {
        PartialAnalysis[End] = combineValues(PartialAnalysis[End],
                                             UpdatedEndAnalysis);
        for (auto Node : successors(End)) {
          Worklist.push({ End, Node });
        }
      }
    }

    // Step 3 presenting the results
    for (auto &[Node, Analysis] : PartialAnalysis) {
      AnalysisResult[Node] = { PartialAnalysis[Node],
                               applyTransferFunction(Node,
                                                     PartialAnalysis[Node]) };
    }
    return AnalysisResult;
  }

private:
  static auto successors(Label From) {
    return llvm::make_range(GT::child_begin(From), GT::child_end(From));
  }
};

} // namespace TypeShrinking
