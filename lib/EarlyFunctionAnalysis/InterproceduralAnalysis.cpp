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

bool ABI::isLessOrEqual(const ABI &RHS) const {
  if (SUL::isLessOrEqual(Arguments, RHS.Arguments)
      and SUL::isLessOrEqual(Returns, RHS.Returns)) {
    return true;
  }
  return false;
}

ABI ABI::combineValues(const ABI &R) const {
  ABI Result;
  Result.Arguments = SUL::combineValues(Arguments, R.Arguments);
  Result.Returns = SUL::combineValues(Returns, R.Returns);

  return Result;
}

InterproceduralLattice
InterproceduralAnalysis::combineValues(const InterproceduralLattice &E1,
                                       const InterproceduralLattice &E2) const {
  InterproceduralLattice Result;

  for (auto &[L, R] : zipmap_range(E1.FinalABI, E2.FinalABI)) {
    Result.FinalABI[L->first] = L->second.combineValues(R->second);
  }

  return Result;
}

bool InterproceduralAnalysis::isLessOrEqual(const InterproceduralLattice &E1,
                                            const InterproceduralLattice &E2) const {
  if (E2.IsExtremal) {
    return true;
  }

  for (auto &[K, V] : zipmap_range(E1.FinalABI, E2.FinalABI)) {
    if (K == nullptr or V == nullptr)
      continue;

    if (!K->second.isLessOrEqual(V->second)) {
      return false;
    }
  }

  return true;
}

InterproceduralLattice
InterproceduralAnalysis::applyTransferFunction(Label L,
                                               const InterproceduralLattice &E2) const {
  using namespace ABIAnalyses;
  if (L == nullptr) {
    return E2;
  }

  if (std::holds_alternative<Node::AnalysisType>(L->Type)) {
    InterproceduralLattice Result = E2;
    // std::map<const llvm::GlobalVariable *, abi::RegisterState::Values>
    // ABIResults;
    const auto *BB = L->BB;
    revng_assert(BB != nullptr);
    switch (std::get<Node::AnalysisType>(L->Type)) {
    case Node::AnalysisType::UAOF: {
      revng_assert(L->isFunction());
      const auto &EntryAddress = L->getFunctionAddress();

      // copy results of UAOF for BB
      const auto &Results = PartialResults.at(EntryAddress);
      for (const llvm::BasicBlock *Next : llvm::depth_first(BB)) {
        for (const llvm::Instruction &I : *Next) {
          if (auto *Call = getCallTo(&I, PreCallHook)) {
            auto PC = BasicBlockID::fromValue(Call->getArgOperand(0));
            copyResults(Results.CallSites.at(PC).ArgumentsRegisters, Result.FinalABI.at(EntryAddress).Arguments);
          } else if (isCallTo(&I, PostCallHook)) {
            // copy returns of Next
          } else if (isCallTo(&I, RetHook)) {
            // copy returns of Next
          }
        }
      }
      break;
    }
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

} // namespace efa
