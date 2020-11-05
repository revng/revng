#pragma once

#include <functional>
#include <map>
#include <queue>
#include <type_traits>
#include <unordered_map>

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

template<class LatticeElement, class T>
struct MonotoneFramework {
  using Graph = llvm::GraphTraits<T>;
  // by the specs of llvm::GraphTraits, NodeRef chould be cheap to copy
  using Label = typename Graph::NodeRef;

  using LatticeElementPair = std::pair<LatticeElement, LatticeElement>;

  LatticeElement combineValues(const LatticeElement &, const LatticeElement);
  LatticeElement applyTransferFunction(Label, const LatticeElement &);
  bool isLessOrEqual(const LatticeElement &, const LatticeElement);

  /// Compute the maximum fixed points of an instance of monotone framework
  std::map<Label, LatticeElementPair>
  getMaximalFixedPoint(const Graph &Flow,
                       LatticeElement BottomValue,
                       LatticeElement ExtremalValue,
                       const std::vector<Label> &ExtremalLabels) {
    std::map<Label, LatticeElement> PartialAnalysis;
    std::map<Label, LatticeElementPair> AnalysisResult;
    std::queue<std::pair<Label, Label>> Worklist;

    // Step 1.1 initialize the worklist and extremal labels
    for (auto Start : llvm::nodes(Flow)) {
      PartialAnalysis[Start] = BottomValue;
      for (auto End : llvm::children(Start)) {
        Worklist.push({ Start, End });
      }
    }
    for (auto ExtremalLabel : ExtremalLabels) {
      PartialAnalysis[ExtremalLabel] = ExtremalValue;
    }

    // Step 2 iteration
    while (!Worklist.empty()) {
      auto [Start, End] = Worklist.pop();
      auto &Partial = PartialAnalysis[Start];
      auto UpdatedEndAnalysis = applyTransferFunction(Start, Partial);
      if (!isLessOrEqual(UpdatedEndAnalysis, PartialAnalysis[End])) {
        PartialAnalysis[End] = combineValues(PartialAnalysis[End],
                                             UpdatedEndAnalysis);
        for (auto Node : llvm::children(End)) {
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
};

} // namespace TypeShrinking
