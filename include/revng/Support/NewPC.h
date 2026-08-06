#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/BasicBlockID.h"
#include "revng/Support/IRHelper.h"
#include "revng/Support/MetaAddress.h"

/// The arguments of `newpc`
enum class NewPCArgument {
  InstructionID,
  InstructionSize,
  IsJumpTarget,
  OwnerFunction,
  FirstLocalVariable
};

/// Marks the beginning of the code translated from an instruction
inline IRHelper<NewPCArgument> NewPCHelper("newpc");

inline BasicBlockID blockIDFromNewPC(ConstIRHelperCall<NewPCArgument> Call) {
  auto *Argument = Call.getArgument(NewPCArgument::InstructionID);
  return BasicBlockID::fromValue(Argument);
}

inline MetaAddress addressFromNewPC(ConstIRHelperCall<NewPCArgument> Call) {
  return blockIDFromNewPC(Call).notInlinedAddress();
}

/// \return the entry of the function whose control-flow graph owns the block
///         \p Call belongs to, or an invalid `MetaAddress` in the root
///         function, where code has not been attributed to a function yet
inline MetaAddress ownerFromNewPC(ConstIRHelperCall<NewPCArgument> Call) {
  return MetaAddress::fromValue(Call.getArgument(NewPCArgument::OwnerFunction));
}

/// \return whether the instruction \p Call marks is the first of a basic block
inline bool startsBasicBlock(ConstIRHelperCall<NewPCArgument> Call) {
  return getLimitedValue(Call.getArgument(NewPCArgument::IsJumpTarget)) == 1;
}

/// Record in each `newpc` of \p F that its block belongs to the control-flow
/// graph of \p Owner
void setNewPCOwner(llvm::Function *F, const MetaAddress &Owner);

inline MetaAddress getBasicBlockJumpTarget(llvm::BasicBlock *BB) {
  llvm::Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return MetaAddress::invalid();

  if (std::optional Call = NewPCHelper.getCall(I))
    if (startsBasicBlock(*Call))
      return blockIDFromNewPC(*Call).start();

  return MetaAddress::invalid();
}

inline BasicBlockID getBasicBlockID(const llvm::BasicBlock *BB) {
  revng_assert(BB != nullptr);

  const llvm::Instruction *I = BB->getFirstNonPHI();
  if (I == nullptr)
    return BasicBlockID::invalid();

  if (std::optional Call = NewPCHelper.getCall(I))
    return blockIDFromNewPC(*Call);

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

/// Return the address of the instruction following \p TheInstruction.
inline MetaAddress getNextPC(llvm::Instruction *TheInstruction) {
  auto [PC, Size] = getPC(TheInstruction);
  return PC + Size;
}
