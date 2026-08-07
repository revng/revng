#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/BasicBlockID.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

namespace NewPCArguments {
enum {
  InstructionID,
  InstructionSize,
  IsJumpTarget,
  InliningIndex,
  DissassembledInstruction,
  OwnerFunction,
  FirstLocalVariable
};
} // namespace NewPCArguments

inline MetaAddress getBasicBlockJumpTarget(llvm::BasicBlock *BB) {
  using namespace llvm;

  Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return MetaAddress::invalid();

  if (llvm::CallInst *Call = getCallTo(I, "newpc")) {
    if (getLimitedValue(Call->getOperand(2)) == 1) {
      return MetaAddress::fromValue(Call->getOperand(0));
    }
  }

  return MetaAddress::invalid();
}

inline BasicBlockID blockIDFromNewPC(const llvm::CallBase *Call) {
  revng_assert(isCallTo(Call, "newpc"));
  using namespace NewPCArguments;
  auto *Argument = Call->getArgOperand(InstructionID);
  return BasicBlockID::fromValue(Argument);
}

inline BasicBlockID blockIDFromNewPC(const llvm::Instruction *I) {
  return blockIDFromNewPC(llvm::cast<llvm::CallBase>(I));
}

inline MetaAddress addressFromNewPC(const llvm::CallBase *Call) {
  return blockIDFromNewPC(Call).notInlinedAddress();
}
inline MetaAddress addressFromNewPC(const llvm::Instruction *I) {
  return addressFromNewPC(llvm::cast<llvm::CallBase>(I));
}

/// \return the entry of the function whose control-flow graph owns the block
///         \p Call belongs to, or an invalid `MetaAddress` in the root
///         function, where code has not been attributed to a function yet
inline MetaAddress ownerFromNewPC(const llvm::CallBase *Call) {
  revng_assert(isCallTo(Call, "newpc"));
  using namespace NewPCArguments;
  return MetaAddress::fromValue(Call->getArgOperand(OwnerFunction));
}

inline MetaAddress ownerFromNewPC(const llvm::Instruction *I) {
  return ownerFromNewPC(llvm::cast<llvm::CallBase>(I));
}

/// Record in each `newpc` of \p F that its block belongs to the control-flow
/// graph of \p Owner
void setNewPCOwner(llvm::Function *F, const MetaAddress &Owner);

inline BasicBlockID getBasicBlockID(const llvm::BasicBlock *BB) {
  using namespace llvm;

  revng_assert(BB != nullptr);

  const Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return BasicBlockID::invalid();

  if (const llvm::CallInst *Call = getCallTo(I, "newpc"))
    return blockIDFromNewPC(Call);

  return BasicBlockID::invalid();
}

inline MetaAddress getBasicBlockAddress(const llvm::BasicBlock *BB) {
  return getBasicBlockID(BB).notInlinedAddress();
}

/// Find the first call to NewPC starting from \p TheInstruction
///
llvm::CallInst *getLastNewPC(llvm::Instruction *TheInstruction);

/// Find the PC which lead to generated \p TheInstruction
///
/// \return a pair of integers: the first element represents the PC and the
///         second the size of the instruction.
std::pair<MetaAddress, uint64_t> getPC(llvm::Instruction *TheInstruction);
