#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"

#include "revng/Support/Assert.h"

namespace JumpTargetReason {

inline constexpr const char *MDName = "revng.jt.reasons";

/// Reason for registering a jump target.
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
  /// This jump target has been discovered *after* we added all the entry
  /// addresses of model::Functions
  DependsOnModelFunction = SimpleLiteral << 1,
  LastReason = DependsOnModelFunction
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
  case DependsOnModelFunction:
    return "DependsOnModelFunction";
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
  else if (ReasonName == "DependsOnModelFunction")
    return DependsOnModelFunction;
  else
    revng_abort();
}

inline bool hasReason(uint32_t Reasons, Values ToCheck) {
  return (Reasons & static_cast<uint32_t>(ToCheck)) != 0;
}

inline uint32_t getReasons(const llvm::Instruction *Terminator) {
  using namespace llvm;

  revng_assert(Terminator->isTerminator());

  uint32_t Result = 0;
  const auto *Reasons = cast_or_null<MDTuple>(Terminator->getMetadata(MDName));
  revng_assert(Reasons != nullptr);

  for (const Metadata *ReasonMD : Reasons->operands()) {
    StringRef Text = cast<MDString>(ReasonMD)->getString();
    Result |= static_cast<uint32_t>(fromName(Text));
  }

  return Result;
}

inline uint32_t getReasons(const llvm::BasicBlock *BB) {
  return getReasons(BB->getTerminator());
}

} // namespace JumpTargetReason
