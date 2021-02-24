#include <cstdio>
#include <iostream>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/Casting.h>

#include "revng/ABIAnalyses/DeadRegisterArgumentsOfFunction.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"

namespace DeadRegisterArgumentsOfFunction {
using namespace ABIAnalyses;
using LatticeElement = MFI::LatticeElement;

static State getMax(State Lh, State Rh) {
  if (Lh == NoOrDead) {
    return Rh;
  } else if (Lh == Maybe) {
    if (Rh == NoOrDead) {
      return Lh;
    } else {
      return Rh;
    }
  } else {
    return Lh;
  }
}

LatticeElement MFI::combineValues(const LatticeElement &Lh,
                                  const LatticeElement &Rh) const {

  LatticeElement New = Lh;
  for (const auto &KV : Rh) {
    if (New.find(KV.first) != New.end()) {
      New[KV.first] = getMax(New[KV.first], KV.second);
    } else {
      New[KV.first] = KV.second;
    }
  }
  return New;
}

bool MFI::isLessOrEqual(const LatticeElement &Lh,
                        const LatticeElement &Rh) const {
  if (Lh.size() > Rh.size())
    return false;
  for (auto &E : Lh) {
    if (Rh.count(E.first) == 0) {
      return false;
    }
    auto RhState = Rh.lookup(E.first);
    if (getMax(E.second, RhState) != RhState) {
      return false;
    }
  }
  return true;
}

LatticeElement MFI::applyTransferFunction(Label L,
                                          const LatticeElement &E) const {
  LatticeElement New = E;
  for (auto &I : *L) {
    switch (classifyInstruction(&I)) {
    case Read: {
      for (auto &Reg : getRegistersRead(&I)) {
        const auto &V = New.find(Reg);
        if (V == New.end() || V->getSecond() == Maybe) {
          New[Reg] = Unknown;
        }
      }
      break;
    }
    case Write: {
      for (auto &Reg : getRegistersWritten(&I)) {
        const auto &V = New.find(Reg);
        if (V == New.end() || V->getSecond() == Maybe) {
          New[Reg] = NoOrDead;
        }
      }
      break;
    }
    case TheCall:
    case WeakWrite:
    case None:
      break;
    }
  }
  return New;
}

llvm::DenseMap<llvm::GlobalVariable *, State>
analyze(const llvm::Function *F,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {


  MFI Instance{{GCBI, FP}};
  llvm::DenseMap<int32_t, State> InitialValue{};
  llvm::DenseMap<int32_t, State> ExtremalValue{};

  auto Results = MFP::getMaximalFixedPoint<MFI>(
      Instance, &F->getEntryBlock(), InitialValue, ExtremalValue,
      {&F->getEntryBlock()}, {&F->getEntryBlock()});

  llvm::DenseMap<llvm::GlobalVariable *, State> RegUnknown{};
  llvm::DenseMap<llvm::GlobalVariable *, State> RegNoOrDead{};

  for (auto &[BB, Result] : Results) {
    for (auto &[RegID, RegState] : Result.OutValue) {
      if (RegState == Unknown) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          RegUnknown[GV] = Unknown;
        }
      }
    }
  }

  for (auto &[BB, Result] : Results) {
    for (auto &[RegID, RegState] : Result.OutValue) {
      if (RegState == NoOrDead) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          if (RegUnknown.count(GV) == 0) {
            RegNoOrDead[GV] = NoOrDead;
          }
        }
      }
    }
  }

  return RegNoOrDead;
}
} // namespace DeadRegisterArgumentsOfFunction
