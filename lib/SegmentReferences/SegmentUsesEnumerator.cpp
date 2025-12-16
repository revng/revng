//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/ADT/Queue.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"

using namespace llvm;

SegmentUsesEnumerator::UseList
SegmentUsesEnumerator::getUses(Module &M, Function *LimitTo) {
  UseList Result;
  auto &Context = M.getContext();

  // Initialize queue with the uses of segment globals
  OnceQueue<std::pair<MetaAddress, llvm::Use *>> Queue;
  for (llvm::GlobalVariable &SegmentGlobal :
       FunctionTags::SegmentGlobal.globals(&M)) {
    MetaAddress StartAddress = SegmentGlobal::getAddress(SegmentGlobal);

    const model::Segment *SegmentPtr = Binary.getSegmentFor(StartAddress).first;
    revng_assert(SegmentPtr != nullptr);
    const model::Segment &Segment = *SegmentPtr;
    if (SegmentAccess == SegmentAccess::ReadOnly and Segment.IsWriteable())
      continue;

    if (SegmentAccess == SegmentAccess::ExecutableOnly
        and not Segment.IsExecutable())
      continue;

    for (Use &U : SegmentGlobal.uses()) {
      Queue.insert({ StartAddress, &U });
    }
  }

  while (not Queue.empty()) {
    auto [Address, Use] = Queue.pop();
    User *User = Use->getUser();
    auto *I = dyn_cast<Instruction>(User);

    if (I != nullptr and LimitTo != nullptr and I->getFunction() != LimitTo) {

      // If we have a whitelist, ignore instructions not in the whitelist
      continue;

    } else if (shouldSkip(*Use)) {

      for (auto &UserUse : User->uses())
        Queue.insert({ Address, &UserUse });

    } else if (auto *Return = dyn_cast<ReturnInst>(User)) {

      // Handle being returned
      // Note: this enables use to handle transparently SegmentGlobalGetters
      for (CallBase *Call : callers(Return->getFunction()))
        for (auto &UserUse : Call->uses())
          Queue.insert({ Address, &UserUse });

    } else if (auto MaybeAddend = getAddend(*Use);
               Use->getOperandNo() == 0 and MaybeAddend.has_value()) {

      // Handle add

      // Compute address
      auto Offset = MaybeAddend.value();
      auto CandidateAddress = Address + Offset;
      const model::Segment *SegmentPtr = Binary.getSegmentFor(Address).first;
      revng_assert(SegmentPtr != nullptr);

      // Check if we're out of the segment
      const model::Segment &Segment = *SegmentPtr;
      if (not Segment.contains(CandidateAddress))
        continue;

      for (auto &UserUse : User->uses())
        Queue.insert({ CandidateAddress, &UserUse });

    } else if (I != nullptr) {
      Result.emplace_back(Use, Address);
    }
  }

  return Result;
}

unsigned int SegmentUsesEnumerator::getOpcode(User &User) {
  if (auto *CE = dyn_cast<ConstantExpr>(&User)) {
    return CE->getOpcode();
  } else if (auto *I = dyn_cast<Instruction>(&User)) {
    return I->getOpcode();
  } else {
    return 0;
  }
}

bool SegmentUsesEnumerator::shouldSkip(Use &TheUse) {
  User *User = TheUse.getUser();
  unsigned int Opcode = getOpcode(*User);

  // Skip over inttoptr and ptrtoint
  if (Opcode == llvm::Instruction::IntToPtr
      or Opcode == llvm::Instruction::PtrToInt) {
    return true;
  }

  return false;
}

std::optional<uint64_t> SegmentUsesEnumerator::getAddend(Use &TheUse) {
  User *User = TheUse.getUser();
  unsigned int Opcode = getOpcode(*User);
  bool IsAdd = Opcode == llvm::Instruction::Add;
  bool IsSub = Opcode == llvm::Instruction::Sub;

  if (not IsAdd and not IsSub)
    return std::nullopt;

  auto *CI = dyn_cast<ConstantInt>(User->getOperand(1));
  if (CI == nullptr)
    return std::nullopt;

  llvm::APInt Result;
  if (IsAdd) {
    Result = CI->getValue();
  } else if (IsSub) {
    Result = -CI->getValue();
  }

  if (Result.getBitWidth() > 64 or Result.getBitWidth() == 1)
    return std::nullopt;

  return Result.getLimitedValue();
}
