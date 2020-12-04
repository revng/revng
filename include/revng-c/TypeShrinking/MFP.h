#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <map>
#include <queue>
#include <type_traits>

#include "llvm/ADT/GraphTraits.h"

#include "revng/ADT/GenericGraph.h"

namespace TypeShrinking {

template<typename T, typename U>
concept same_as = std::is_same_v<T, U>;

template<typename LatticeElement>
struct MFPResult {
  LatticeElement inValue;
  LatticeElement outValue;
};

template<typename GraphType>
auto successors(typename llvm::GraphTraits<GraphType>::NodeRef From) {
  return llvm::make_range(llvm::GraphTraits<GraphType>::child_begin(From),
                          llvm::GraphTraits<GraphType>::child_end(From));
}

template<typename MFI>
concept MonotoneFrameworkInstance = requires(typename MFI::LatticeElement E1,
                                             typename MFI::LatticeElement E2,
                                             typename MFI::Label L) {
  same_as<typename MFI::Label,
          typename llvm::GraphTraits<typename MFI::GraphType>::NodeRef>;
  // Disable clang-format, because it does not handle concepts very well yet
  // clang-format off
  { MFI::combineValues(E1, E2) } ->same_as<typename MFI::LatticeElement>;
  { MFI::isLessOrEqual(E1, E2) } ->same_as<bool>;
  { MFI::applyTransferFunction(L, E2) } ->same_as<typename MFI::LatticeElement>;
  // clang-format on
};

/// Compute the maximum fixed points of an instance of monotone framework
template<MonotoneFrameworkInstance MFI>
std::map<typename MFI::Label, MFPResult<typename MFI::LatticeElement>>
getMaximalFixedPoint(const typename MFI::GraphType &Flow,
                     typename MFI::LatticeElement InitialValue,
                     typename MFI::LatticeElement ExtremalValue,
                     const std::vector<typename MFI::Label> &ExtremalLabels) {
  typedef typename MFI::Label Label;
  typedef typename MFI::LatticeElement LatticeElement;
  std::map<Label, LatticeElement> PartialAnalysis;
  std::map<Label, MFPResult<LatticeElement>> AnalysisResult;
  std::queue<std::pair<Label, Label>> Worklist;

  // Step 1 initialize the worklist and extremal labels
  for (Label ExtremalLabel : ExtremalLabels) {
    PartialAnalysis[ExtremalLabel] = ExtremalValue;
  }
  for (Label Start : llvm::nodes(Flow)) {
    if (PartialAnalysis.find(Start) == PartialAnalysis.end()) {
      PartialAnalysis[Start] = InitialValue;
    }

    for (Label End : successors<typename MFI::GraphType>(Start)) {
      Worklist.push({ Start, End });
    }
  }

  // Step 2 iteration
  while (!Worklist.empty()) {
    auto [Start, End] = Worklist.front();
    Worklist.pop();
    auto &ParialStart = PartialAnalysis.at(Start);
    LatticeElement UpdatedEndAnalysis = MFI::applyTransferFunction(Start,
                                                                   ParialStart);
    auto &PartialEnd = PartialAnalysis.at(End);
    if (!MFI::isLessOrEqual(UpdatedEndAnalysis, PartialEnd)) {
      PartialEnd = MFI::combineValues(PartialEnd, UpdatedEndAnalysis);
      for (Label Node : successors<typename MFI::GraphType>(End)) {
        Worklist.push({ End, Node });
      }
    }
  }

  // Step 3 presenting the results
  for (auto &[Node, Analysis] : PartialAnalysis) {
    AnalysisResult[Node] = {
      PartialAnalysis[Node],
      MFI::applyTransferFunction(Node, PartialAnalysis[Node])
    };
  }
  return AnalysisResult;
}

} // namespace TypeShrinking
