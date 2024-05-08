#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"

#include "revng/Lift/LoadBinaryPass.h"
#include "revng/Support/Debug.h"

namespace JTReason {

// TODO: move me to another header file
/// Reason for registering a jump target
enum Values {
  /// PC after an helper (e.g., a syscall)
  PostHelper = 1,
  /// Obtained from a direct store to the PC
  DirectJump = 2,
  /// Obtained digging in global data
  GlobalData = 4,
  /// Fallthrough of multiple instructions in the immediately preceding bytes
  AmbiguousInstruction = 8,
  /// Stored in the PC
  PCStore = 16,
  /// Stored in memory
  MemoryStore = 32,
  /// Obtained digging in global data, but never used. Likely a function
  /// pointer
  UnusedGlobalData = 64,
  /// This JT is the target of a call instruction.
  Callee = 128,
  /// A load has been performed from this address
  LoadAddress = 256,
  /// Obtained as the fallthrough of a function call
  ReturnAddress = 512,
  /// Obtained from a function symbol
  FunctionSymbol = 1024,
  /// Immediate value in the IR, usually a return address
  SimpleLiteral = 2048,
  LastReason = SimpleLiteral
};

inline const char *getName(Values Reason) {
  switch (Reason) {
  case PostHelper:
    return "PostHelper";
  case DirectJump:
    return "DirectJump";
  case GlobalData:
    return "GlobalData";
  case AmbiguousInstruction:
    return "AmbiguousInstruction";
  case PCStore:
    return "PCStore";
  case MemoryStore:
    return "MemoryStore";
  case UnusedGlobalData:
    return "UnusedGlobalData";
  case Callee:
    return "Callee";
  case LoadAddress:
    return "LoadAddress";
  case ReturnAddress:
    return "ReturnAddress";
  case FunctionSymbol:
    return "FunctionSymbol";
  case SimpleLiteral:
    return "SimpleLiteral";
  }

  revng_abort();
}

inline Values fromName(llvm::StringRef ReasonName) {
  if (ReasonName == "PostHelper")
    return PostHelper;
  else if (ReasonName == "DirectJump")
    return DirectJump;
  else if (ReasonName == "GlobalData")
    return GlobalData;
  else if (ReasonName == "AmbiguousInstruction")
    return AmbiguousInstruction;
  else if (ReasonName == "PCStore")
    return PCStore;
  else if (ReasonName == "MemoryStore")
    return MemoryStore;
  else if (ReasonName == "UnusedGlobalData")
    return UnusedGlobalData;
  else if (ReasonName == "Callee")
    return Callee;
  else if (ReasonName == "LoadAddress")
    return LoadAddress;
  else if (ReasonName == "ReturnAddress")
    return ReturnAddress;
  else if (ReasonName == "FunctionSymbol")
    return FunctionSymbol;
  else if (ReasonName == "SimpleLiteral")
    return SimpleLiteral;
  else
    revng_abort();
}

inline bool hasReason(uint32_t Reasons, Values ToCheck) {
  return (Reasons & static_cast<uint32_t>(ToCheck)) != 0;
}

} // namespace JTReason

namespace KillReason {

enum Values {
  NonKiller,
  KillerSyscall,
  EndlessLoop,
  LeadsToKiller
};

inline llvm::StringRef getName(Values Reason) {
  switch (Reason) {
  case NonKiller:
    return "NonKiller";
  case KillerSyscall:
    return "KillerSyscall";
  case EndlessLoop:
    return "EndlessLoop";
  case LeadsToKiller:
    return "LeadsToKiller";
  }

  revng_abort("Unexpected reason");
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "NonKiller")
    return NonKiller;
  if (Name == "KillerSyscall")
    return KillerSyscall;
  else if (Name == "EndlessLoop")
    return EndlessLoop;
  else if (Name == "LeadsToKiller")
    return LeadsToKiller;
  else
    revng_abort("Unexpected name");
}

} // namespace KillReason

class LiftPass : public llvm::ModulePass {
public:
  static char ID;

public:
  LiftPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadBinaryWrapperPass>();
    AU.addRequired<LoadModelWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};
