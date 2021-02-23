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

TransferKind MFI::classifyInstruction(const llvm::Instruction *I) const {
  switch (I->getOpcode()) {
  case llvm::Instruction::Store: {
    auto S = llvm::cast<llvm::StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      return Write;
    }
    break;
  }
  case llvm::Instruction::Load: {
    auto L = llvm::cast<llvm::LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      return Read;
    }
    break;
  }
  case llvm::Instruction::Call: {
    if (I == CallSite) {
      return TheCall;
    }
    return WeakWrite;
  }
  }
  return None;
}

llvm::SmallVector<int32_t, 1>
MFI::getRegistersWritten(const llvm::Instruction *I) const {
  llvm::SmallVector<int32_t, 1> Result;
  switch (I->getOpcode()) {
  case llvm::Instruction::Store: {
    auto S = llvm::cast<llvm::StoreInst>(I);
    if (isABIRegister(S->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        llvm::cast<llvm::GlobalVariable>(S->getPointerOperand())));
    }
    break;
  }
  }
  return Result;
}

llvm::SmallVector<int32_t, 1>
MFI::getRegistersRead(const llvm::Instruction *I) const {
  llvm::SmallVector<int32_t, 1> Result;
  switch (I->getOpcode()) {
  case llvm::Instruction::Load: {
    auto L = llvm::cast<llvm::LoadInst>(I);
    if (isABIRegister(L->getPointerOperand())) {
      Result.push_back(ABIRegisters.lookup(
        llvm::cast<llvm::GlobalVariable>(L->getPointerOperand())));
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
  for (auto &I : *L) {
    switch (classifyInstruction(&I)) {
    case TheCall: {
      if (E.count(InitialRegisterState) != 0) {
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
        const auto &V = E.find(Reg);
        if (V == E.end() || V->getSecond() == Maybe) {
          New[Reg] = Unknown;
        }
      }
      break;
    }
    case Write: {
      for (auto &Reg : getRegistersWritten(&I)) {
        const auto &V = E.find(Reg);
        if (V == E.end() || V->getSecond() == Maybe) {
          New[Reg] = NoOrDead;
        }
      }
      break;
    }
    case WeakWrite:
    case None:
      break;
    }
  }
  New.erase(InitialRegisterState);
  return New;
}

llvm::DenseMap<llvm::GlobalVariable *, State>
analyze(const llvm::Instruction *CallSite,
        llvm::Function *Entry,
        const GeneratedCodeBasicInfo &GCBI,
        const StackAnalysis::FunctionProperties &FP) {
  llvm::errs() << "going " << Entry->getName().str() << " " << Entry->size()
               << '\n';
  CallSite->print(llvm::errs());
  llvm::errs() << '\n';
  llvm::DenseMap<llvm::GlobalVariable *, int32_t> ABIRegisters;
  llvm::DenseMap<int32_t, llvm::GlobalVariable *> IndexABIRegisters;

  for (auto *CSV : GCBI.csvs())
    if (CSV) {
      ABIRegisters[CSV] = ABIRegisters.size();
      IndexABIRegisters[ABIRegisters[CSV]] = CSV;
    }

  MFI Instance{ CallSite, ABIRegisters, FP };
  llvm::DenseMap<int32_t, State> InitialValue{ { InitialRegisterState,
                                                 Unknown } };
  llvm::DenseMap<int32_t, State> ExtremalValue{ { InitialRegisterState,
                                                  Unknown } };

  auto Results = MFP::getMaximalFixedPoint<MFI>(Instance,
                                                CallSite->getParent(),
                                                InitialValue,
                                                ExtremalValue,
                                                { CallSite->getParent() },
                                                { CallSite->getParent() });
  llvm::errs() << "finished MFP " << Results.size() << "\n";
  llvm::DenseMap<llvm::GlobalVariable *, State> RegNoOrDead{};
  // start debug stuff

  for (auto &Result : Results) {
    Result.first->print(llvm::errs());
    llvm::errs() << '\n';
    for (auto &KV : Result.second.OutValue) {
      llvm::errs() << "BB RAW " << KV.first << " " << KV.second << "\n";
      if (IndexABIRegisters.count(KV.first) != 0) {
        auto *GV = IndexABIRegisters[KV.first];
        llvm::errs() << "BB RESULT " << GV->getName().str() << " " << KV.second
                     << "\n";
      }
    }
  }
  // end debug stuff
  for (auto &Result : Results) {
    for (auto &KV : Result.second.OutValue) {
      if (KV.second == NoOrDead && IndexABIRegisters.count(KV.first) != 0) {
        auto *GV = IndexABIRegisters[KV.first];
        if (Instance.isABIRegister(GV)) {
          RegNoOrDead[GV] = NoOrDead;
        }
      }
    }
  }
  return RegNoOrDead;
} 
} // namespace DeadReturnValuesOfFunctionCall
