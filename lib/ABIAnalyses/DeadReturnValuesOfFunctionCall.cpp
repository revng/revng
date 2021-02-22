//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/ABIAnalyses/DeadReturnValuesOfFunctionCall.h"
#include "revng/MFP/MFP.h"
#include "revng/Support/revng.h"

namespace DeadReturnValuesOfFunctionCall {
using namespace llvm;
using LatticeElement = MFI::LatticeElement;

bool MFI::isABIRegister(const Value *V) const {
  if (const auto &G = dyn_cast<GlobalVariable>(V)) {
    if (ABIRegisters.count(G) != 0) {
      return true;
    }
  }
  return false;
}

TransferKind MFI::classifyInstruction(const Instruction &I) const {
  switch (I.getOpcode()) {
  case Instruction::Store: {
    auto &S = cast<StoreInst>(I);
    if (isABIRegister(S.getPointerOperand())) {
      return Write;
    }
    break;
  }
  case Instruction::Load: {
    auto &L = cast<LoadInst>(I);
    if (isABIRegister(L.getPointerOperand())) {
      return Read;
    }
    break;
  }
  case Instruction::Call: {
    auto &C = cast<CallInst>(I);
    if (C.getCalledFunction() == I.getFunction()) {
      return TheCall;
    }
    return WeakWrite;
  }
  }
  return None;
}

SmallVector<int32_t, 1> MFI::getRegistersWritten(const Instruction &I) const {
  SmallVector<int32_t, 1> Result;
  switch (I.getOpcode()) {
  case Instruction::Store: {
    auto &S = cast<StoreInst>(I);
    if (isABIRegister(S.getPointerOperand())) {
      Result.push_back(
        ABIRegisters.lookup(cast<GlobalVariable>(S.getPointerOperand())));
    }
    break;
  }
  case Instruction::Call: {
    auto &C = cast<CallInst>(I);
    if (C.getCalledFunction() != I.getFunction()) {
      for (auto &Reg : Properties.getRegistersClobbered(
             &C.getCalledFunction()->getEntryBlock())) {
        Result.push_back(ABIRegisters.lookup(cast<GlobalVariable>(Reg)));
      }
    }
  }
  }
  return Result;
}

SmallVector<int32_t, 1> MFI::getRegistersRead(const Instruction &I) const {
  SmallVector<int32_t, 1> Result;
  switch (I.getOpcode()) {
  case Instruction::Load: {
    auto &L = cast<LoadInst>(I);
    if (isABIRegister(L.getPointerOperand())) {
      Result.push_back(
        ABIRegisters.lookup(cast<GlobalVariable>(L.getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}

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

LatticeElement
MFI::combineValues(const LatticeElement &Lh, const LatticeElement &Rh) const {

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

  return false;
}

LatticeElement
MFI::applyTransferFunction(Label L, const LatticeElement &E) const {
  LatticeElement New = E;
  for (auto &I : *L) {
    switch (classifyInstruction(I)) {
    case TheCall:
    // TODO: for the call we should record which registers are read in the
    // current function?
    case Read: {
      for (auto &Reg : getRegistersRead(I)) {
        const auto &V = E.find(Reg);
        if (V == E.end() || V->getSecond() == Maybe) {
          New[Reg] = Unknown;
        }
      }
    }
    case Write: {
      for (auto &Reg : getRegistersWritten(I)) {
        const auto &V = E.find(Reg);
        if (V == E.end() || V->getSecond() == Maybe) {
          New[Reg] = NoOrDead;
        }
      }
    }
    case WeakWrite:
    case None:
      break;
    }
  }
  return E;
}

static std::vector<const BasicBlock *> getCallSites(const Function *F) {
  std::vector<const BasicBlock *> CallSites{};
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (I.getOpcode() == Instruction::Call) {
        CallSites.push_back(&BB);
      }
    }
  }
  return CallSites;
}

inline std::map<MFI::Label, MFP::MFPResult<LatticeElement>>
analyze(Function *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {
  DenseMap<GlobalVariable *, int32_t> ABIRegisters;
  for (auto *CSV : GCBI.abiRegisters())
    if (CSV && !(GCBI.isSPReg(CSV)))
      ABIRegisters[CSV] = ABIRegisters.size();

  MFI Instance{ ABIRegisters, FP };
  DenseMap<int32_t, State> InitialValue;
  DenseMap<int32_t, State> ExtremalValue;
  for (auto &R : Instance.ABIRegisters) {
    ExtremalValue[R.second] = Unknown;
  }
  auto Result = MFP::getMaximalFixedPoint<MFI>(Instance,
                                               Entry,
                                               InitialValue,
                                               ExtremalValue,
                                               getCallSites(Entry));
  return Result;
}
} // namespace DeadReturnValuesOfFunctionCall
