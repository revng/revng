#pragma once

#include <functional>
#include <type_traits>
#include <unordered_map>

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

/// Compute the maximum fixed points of an instance of monotone framework
template<class LatticeElement, class Label>
std::unordered_map<Label *, std::tuple<LatticeElement, LatticeElement>>
getMaximalFixedPoint(LatticeElement (*combineValues)(LatticeElement &,
                                                     LatticeElement &),
                     bool (*isLessOrEqual)(LatticeElement &, LatticeElement &),
                     GenericGraph<Label> &Flow,
                     LatticeElement ExtremalValue,
                     LatticeElement BottomValue,
                     std::vector<Label *> &ExtremalLabels,
                     // Temporarily disable clang-format here. It conflicts with
                     // revng conventions
                     // clang-format off
                     std::function<LatticeElement(LatticeElement &)>
                       (*getTransferFunction)(Label *)
                     // clang-format on
) {
  static_assert(std::is_base_of_v<ForwardNode<Label>, Label>);

  // Step 1 initialize the worklist and the partial analysis results
  std::unordered_map<Label *, LatticeElement> PartialAnalysis;
  std::unordered_map<Label *, std::tuple<LatticeElement, LatticeElement>>
    AnalysisResult;
  std::deque<std::tuple<Label *, Label *>> Worklist;
  for (auto *Start : Flow.nodes()) {
    PartialAnalysis[Start] = BottomValue;
    for (auto *End : Start->successors()) {
      Worklist.push_back({ Start, End });
    }
  }
  for (auto &ExtremalLabel : ExtremalLabels) {
    PartialAnalysis[ExtremalLabel] = ExtremalValue;
  }

  // Step 2 Iteration
  while (!Worklist.empty()) {
    auto [Start, End] = Worklist.front();
    Worklist.pop_front();
    auto &Partial = PartialAnalysis[Start];
    auto UpdatedEndAnalysis = getTransferFunction(Start)(Partial);
    if (!isLessOrEqual(UpdatedEndAnalysis, PartialAnalysis[End])) {
      PartialAnalysis[End] = combineValues(PartialAnalysis[End],
                                           UpdatedEndAnalysis);
      for (auto Node : End->successors()) {
        Worklist.push_back({ End, Node });
      }
    }
  }

  // Step 3 presenting the results
  for (auto &[Node, Analysis] : PartialAnalysis) {
    AnalysisResult[Node] = { PartialAnalysis[Node],
                             getTransferFunction(Node)(PartialAnalysis[Node]) };
  }
  return AnalysisResult;
}

} // namespace TypeShrinking
