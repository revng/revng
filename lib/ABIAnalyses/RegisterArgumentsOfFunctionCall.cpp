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

#include "revng/ABIAnalyses/RegisterArgumentsOfFunctionCall.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace RegisterArgumentsOfFunctionCall {
using namespace llvm;
using namespace ABIAnalyses;
using LatticeElement = MFI::LatticeElement;

static State getMax(State Lh, State Rh) {
  if (Lh == Bottom) {
    return Rh;
  } else if (Lh == Maybe || Rh == Yes) {
    if (Rh == Bottom) {
      return Lh;
    } else if (Rh == Lh) {
      return Lh;
    } else {
      return Unknown;
    }
  } else {
    return Lh;
  }
}

LatticeElement
MFI::combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const {

  LatticeElement New = Lh;
  for (const auto &KV : Rh) {
    if (New.find(KV.first) != New.end()) {
      New[KV.first] = getMax(New[KV.first], KV.second);
    } else {
      // in other analyses we have a linear lattice
      // here we have a diamond with default value maybe
      New[KV.first] = getMax(Maybe, KV.second);
    }
  }
  New.erase(InitialRegisterState);
  return New;
}

bool MFI::isLessOrEqual(const LatticeElement &Lh,
                        const LatticeElement &Rh) const {
  if (Lh.size() > Rh.size())
    return false;
  for (auto &E : Lh) {
    if (E.first == InitialRegisterState) {
      continue;
    }
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

LatticeElement
MFI::applyTransferFunction(Label L, const LatticeElement &E) const {
  LatticeElement New = E;
  for (auto &I : make_range(L->rbegin(), L->rend())) {
    switch (classifyInstruction(&I)) {
    case TheCall: {
      if (New.count(InitialRegisterState) != 0) {
        // The first time we hit TheCall is actually the start of our graph
        New.clear();
      } else {
        New.clear();
        // This is not the first time we hit TheCall, so there is a path from
        // TheCall to itself, we should return Unknown in this case
        for (auto &Reg : ABIRegisters) {
          New[Reg.second] = Unknown;
        }
      }
      break;
    }
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
          New[Reg] = Yes;
        }
      }
      break;
    }
    default:
      break;
    }
  }
  New.erase(InitialRegisterState);
  return New;
}

DenseMap<GlobalVariable *, State>
analyze(const Instruction *CallSite,
        Function *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {

  MFI Instance{ { CallSite, GCBI, FP } };
  DenseMap<int32_t, State> InitialValue{ { InitialRegisterState, Unknown } };
  DenseMap<int32_t, State> ExtremalValue{ { InitialRegisterState, Unknown } };

  auto Results = MFP::getMaximalFixedPoint<MFI>(Instance,
                                                CallSite->getParent(),
                                                InitialValue,
                                                ExtremalValue,
                                                { CallSite->getParent() },
                                                { CallSite->getParent() });

  DenseMap<GlobalVariable *, State> RegUnknown{};
  DenseMap<GlobalVariable *, State> RegYes{};

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
      if (RegState == Yes) {
        if (auto *GV = Instance.getABIRegister(RegID)) {
          if (RegUnknown.count(GV) == 0) {
            RegYes[GV] = RegState;
          }
        }
      }
    }
  }

  return RegYes;
}
} // namespace RegisterArgumentsOfFunctionCall
