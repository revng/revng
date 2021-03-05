//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/ABIAnalyses/Common.h"
#include "revng/ABIAnalyses/Analyses.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace ABIAnalyses::RegisterArgumentsOfFunctionCall {
using namespace llvm;
using namespace ABIAnalyses;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *CallSiteBlock,
        const GeneratedCodeBasicInfo &GCBI) {
  using MFI = MFI<false>;

  MFI Instance{ { getPostCallHook(CallSiteBlock), GCBI } };
  MFI::LatticeElement InitialValue{};
  MFI::LatticeElement ExtremalValue{};

  auto *Start = CallSiteBlock->getUniqueSuccessor();
  auto Results = MFP::getMaximalFixedPoint<MFI>(Instance,
                                                Start,
                                                InitialValue,
                                                ExtremalValue,
                                                { Start },
                                                { Start });

  DenseMap<const GlobalVariable *, State> RegUnknown{};
  DenseMap<const GlobalVariable *, State> RegYes{};

  for (auto &[BB, Result] : Results) {
    for (auto &[GV, RegState] : Result.OutValue) {
      if (RegState == CoreLattice::Unknown) {
        RegUnknown[GV] = State::Unknown;
      }
    }
  }

  for (auto &[BB, Result] : Results) {
    for (auto &[GV, RegState] : Result.OutValue) {
      if (RegState == CoreLattice::Yes) {
        RegYes[GV] = State::Yes;
      }
    }
  }

  return RegYes;
}
} // namespace ABIAnalyses::RegisterArgumentsOfFunctionCall
