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

#include "revng/ABIAnalyses/Generated/UsedReturnValuesOfFunction.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace UsedReturnValuesOfFunction {
using namespace llvm;

DenseMap<GlobalVariable *, State>
analyze(const Instruction *Return,
        const BasicBlock *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {

  MFI<false> Instance{ { GCBI } };
  DenseMap<Register, CoreLattice::LatticeElement> InitialValue{};
  DenseMap<Register, CoreLattice::LatticeElement> ExtremalValue{};

  auto Results = MFP::getMaximalFixedPoint<MFI<false>>(Instance,
                                                Entry,
                                                InitialValue,
                                                ExtremalValue,
                                                { Entry },
                                                { Entry });

  DenseMap<GlobalVariable *, State> RegUnknown{};
  DenseMap<GlobalVariable *, State> RegYesOrDead{};

  for (auto &[BB, Result] : Results) {
    for (auto &[RegID, RegState] : Result.OutValue) {
      if (RegState == CoreLattice::Unknown) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          RegUnknown[GV] = State::Unknown;
        }
      }
    }
  }

  for (auto &[BB, Result] : Results) {
    for (auto &[RegID, RegState] : Result.OutValue) {
      if (RegState == CoreLattice::YesOrDead) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          if (RegUnknown.count(GV) == 0) {
            RegYesOrDead[GV] = State::YesOrDead;
          }
        }
      }
    }
  }

  return RegYesOrDead;
}
} // namespace UsedReturnValuesOfFunction
