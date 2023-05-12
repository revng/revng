/// \file InterproceduralAnalysis.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/InstIterator.h"

#include "revng/EarlyFunctionAnalysis/ABIAnalysis.h"
#include "revng/EarlyFunctionAnalysis/Outliner.h"

#include "ABIAnalyses/Analyses.h"
#include "InterproceduralAnalysis.h"

namespace efa {

using IA = InterproceduralAnalysis;
using IALE = IA::LatticeElement;

InterproceduralAnalysis::LatticeElement
InterproceduralAnalysis::combineValues(const LatticeElement &E1,
                                       const LatticeElement &E2) const {
  LatticeElement Result;
  Result.Arguments = LatticeElement::SUL::combineValues(E1.Arguments,
                                                        E2.Arguments);
  Result.Returns = LatticeElement::SUL::combineValues(E1.Returns, E2.Returns);

  return Result;
}

bool InterproceduralAnalysis::isLessOrEqual(const LatticeElement &E1,
                                            const LatticeElement &E2) const {
  if (E2.IsExtremal) {
    return true;
  }

  if (LatticeElement::SUL::isLessOrEqual(E1.Arguments, E2.Arguments)) {
    return LatticeElement::SUL::isLessOrEqual(E1.Returns, E2.Returns);
  }

  return false;
}

InterproceduralAnalysis::LatticeElement
InterproceduralAnalysis::applyTransferFunction(Label L,
                                               const LatticeElement &E2) const {
  if (L == nullptr) {
    return E2;
  }

  if (std::holds_alternative<Node::AnalysisType>(L->Type)) {
    LatticeElement Result = E2;
    // std::map<const llvm::GlobalVariable *, abi::RegisterState::Values>
    // ABIResults;
    const auto *BB = L->BB;
    revng_assert(BB != nullptr);
    switch (std::get<Node::AnalysisType>(L->Type)) {
    case Node::AnalysisType::UAOF:
      //for (llvm::Instruction *I : llvm::depth_first(BB)) {
        // if (auto *C = getCallTo(I, PreCallHook)) {
        //   
        // } else if (isCallTo(I, PostCallHook)) {

        // } else if (isCallTo(I, RetHook)) {

        // }
      //}
      break;

    case Node::AnalysisType::RAOFC:
      // ABIResults = ABIAnalyses::RegisterArgumentsOfFunctionCall::analyze(L->BB,
      //                                                                    GCBI);
      // applyResults(Node::ResultType::Arguments, Result, ABIResults);
      break;

    case Node::AnalysisType::URVOF:
      // ABIResults = ABIAnalyses::UsedReturnValuesOfFunction::analyze(BB, GCBI);
      // applyResults(Node::ResultType::Returns, Result, ABIResults);
      break;

    case Node::AnalysisType::URVOFC:
      // ABIResults = ABIAnalyses::UsedReturnValuesOfFunctionCall::analyze(BB,
      //                                                                   GCBI);
      // applyResults(Node::ResultType::Returns, Result, ABIResults);
      break;
    }
    return Result;

  } else {
    return E2;
  }
}

void InterproceduralAnalysis::applyResults(Node::ResultType Type,
                                           LatticeElement &Result,
                                           const ABIMap &ABIResults) const {

  auto &R = (Type == Node::ResultType::Arguments) ? Result.Arguments :
                                                    Result.Returns;
  for (auto &[CSV, State] : ABIResults) {
    if (State == abi::RegisterState::Yes) {
      R.insert(CSV);
    }
  }
}

} // namespace efa
