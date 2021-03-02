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

#include "revng/ABIAnalyses/Common.h"
#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"
#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCallLattice.h"
#include "revng/MFP/MFP.h"
#include "revng/Model/Binary.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {
using namespace llvm;
using namespace ABIAnalyses;
using LatticeElement = MFI::LatticeElement;

LatticeElement
MFI::combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const {
  return ABIAnalyses::combineValues<MFI, CoreLattice>(Lh, Rh);
}

bool MFI::isLessOrEqual(const LatticeElement &Lh,
                        const LatticeElement &Rh) const {
  return ABIAnalyses::isLessOrEqual<MFI, CoreLattice>(Lh, Rh);
}

LatticeElement
MFI::applyTransferFunction(Label L, const LatticeElement &E) const {
  LatticeElement New = E;
  for (auto &I : *L) {
    TransferKind T = classifyInstruction(&I);
    switch (T) {
    case TheCall: {
      for (auto &Reg : getRegisters()) {
        New[Reg] = CoreLattice::transfer(TheCall, New[Reg]);
      }
      break;
    }
    case Read:
      for (auto &Reg : getRegistersRead(&I)) {
        New[Reg] = CoreLattice::transfer(T, New[Reg]);
      }
      break;
    case WeakWrite:
    case Write:
      for (auto &Reg : getRegistersWritten(&I)) {
        New[Reg] = CoreLattice::transfer(T, New[Reg]);
      }
      break;
    default:
      break;
    }
  }
  return New;
}

DenseMap<GlobalVariable *, State>
analyze(const Instruction *CallSite,
        Function *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {

  MFI Instance{ { CallSite, GCBI } };
  DenseMap<Register, State> InitialValue{};
  DenseMap<Register, State> ExtremalValue{};

  auto Results = MFP::getMaximalFixedPoint<MFI>(Instance,
                                                CallSite->getParent(),
                                                InitialValue,
                                                ExtremalValue,
                                                { CallSite->getParent() },
                                                { CallSite->getParent() });

  DenseMap<GlobalVariable *, State> RegNoOrDead{};

  for (auto &[BB, Result] : Results) {
    for (auto &[RegID, RegState] : Result.OutValue) {
      if (RegState == State::NoOrDead) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          RegNoOrDead[GV] = State::NoOrDead;
        }
      }
    }
  }
  return RegNoOrDead;
}
} // namespace DeadReturnValuesOfFunctionCall
