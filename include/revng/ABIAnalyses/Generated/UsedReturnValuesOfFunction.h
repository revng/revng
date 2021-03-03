#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file has been automatically generated from scripts/monotone-framework-lattice.py, please don't change it
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Value.h"

#include "revng/ABIAnalyses/Common.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/MFP/MFP.h"
#include "revng/Model/Binary.h"
#include "revng/StackAnalysis/StackAnalysis.h"
#include "revng/Support/revng.h"

namespace UsedReturnValuesOfFunction {
using namespace ABIAnalyses;
using Register = model::Register::Values;
using State = model::RegisterState::Values;

struct CoreLattice {

// using LatticeElement = model::RegisterState::Values;

enum LatticeElement {
  Bottom,
    Maybe,
    Unknown,
    YesOrDead
};



using TransferFunction = ABIAnalyses::TransferKind;

static bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) {
  return Lh == Rh
    || (Lh == LatticeElement::Bottom && Rh == LatticeElement::Maybe)
    || (Lh == LatticeElement::Bottom && Rh == LatticeElement::Unknown)
    || (Lh == LatticeElement::Bottom && Rh == LatticeElement::YesOrDead)
    || (Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
    || (Lh == LatticeElement::YesOrDead && Rh == LatticeElement::Unknown);
}



static LatticeElement combineValues(const LatticeElement &Lh, const LatticeElement &Rh) {
  if ((Lh == LatticeElement::Bottom && Rh == LatticeElement::Maybe)
      || (Lh == LatticeElement::Maybe && Rh == LatticeElement::Bottom)) {
    return LatticeElement::Maybe;
  } else if ((Lh == LatticeElement::Bottom && Rh == LatticeElement::Unknown)
             || (Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
             || (Lh == LatticeElement::Maybe && Rh == LatticeElement::YesOrDead)
             || (Lh == LatticeElement::Unknown && Rh == LatticeElement::Bottom)
             || (Lh == LatticeElement::Unknown && Rh == LatticeElement::Maybe)
             || (Lh == LatticeElement::Unknown && Rh == LatticeElement::YesOrDead)
             || (Lh == LatticeElement::YesOrDead && Rh == LatticeElement::Maybe)
             || (Lh == LatticeElement::YesOrDead && Rh == LatticeElement::Unknown)) {
    return LatticeElement::Unknown;
  } else if ((Lh == LatticeElement::Bottom && Rh == LatticeElement::YesOrDead)
             || (Lh == LatticeElement::YesOrDead && Rh == LatticeElement::Bottom)) {
    return LatticeElement::YesOrDead;
  }
  return Lh;
}



static LatticeElement transfer(TransferFunction T, const LatticeElement &E) {
  switch(T) {
  case TransferFunction::Read:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::Bottom:
      return LatticeElement::Bottom;
    case LatticeElement::YesOrDead:
      return LatticeElement::YesOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::UnknownFunctionCall:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::Bottom:
      return LatticeElement::Bottom;
    case LatticeElement::YesOrDead:
      return LatticeElement::YesOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::Write:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::YesOrDead;
    case LatticeElement::Bottom:
      return LatticeElement::Bottom;
    case LatticeElement::YesOrDead:
      return LatticeElement::YesOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  default:
    return E;
  }
}



};
template<bool isForward>
struct MFI : ABIAnalyses::ABIAnalysis {
  using LatticeElement = llvm::DenseMap<Register, CoreLattice::LatticeElement>;
  using Label = const llvm::BasicBlock *;
  using GraphType = const llvm::BasicBlock *;

  LatticeElement
  combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const {
    return ABIAnalyses::combineValues<MFI, CoreLattice>(Lh, Rh);
  };
  
  bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) const  {
    return ABIAnalyses::isLessOrEqual<MFI, CoreLattice>(Lh, Rh);
  };

  LatticeElement applyTransferFunction(Label L, const LatticeElement &E) const {
    LatticeElement New = E;
    for (auto &I : (isForward ? (*L) : make_range(L->rbegin(), L->rend()))) {
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
  };
};

llvm::DenseMap<llvm::GlobalVariable *, State>
analyze(const llvm::Instruction *CallSite,
        const llvm::BasicBlock *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP);

} // namespace UsedReturnValuesOfFunction
