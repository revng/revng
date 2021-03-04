//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunction.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace DeadRegisterArgumentsOfFunction {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const Instruction *CallSite,
        const BasicBlock *Entry,
        const GeneratedCodeBasicInfo &GCBI) {
  using MFI = MFI<true>;

  MFI Instance{ { GCBI } };
  MFI::LatticeElement InitialValue{};
  MFI::LatticeElement ExtremalValue{};

  auto Results = MFP::getMaximalFixedPoint<MFI>(Instance,
                                                Entry,
                                                InitialValue,
                                                ExtremalValue,
                                                { Entry },
                                                { Entry });

  DenseMap<const GlobalVariable *, State> RegUnknown{};
  DenseMap<const GlobalVariable *, State> RegNoOrDead{};

  for (auto &[BB, Result] : Results) {
    for (auto &[GV, RegState] : Result.OutValue) {
      RegUnknown[GV] = State::Unknown;
    }
  }

  for (auto &[BB, Result] : Results) {
    for (auto &[GV, RegState] : Result.OutValue) {
      if (RegState == CoreLattice::NoOrDead) {
        RegNoOrDead[GV] = State::NoOrDead;
      }
    }
  }

  return RegNoOrDead;
}
} // namespace DeadRegisterArgumentsOfFunction
