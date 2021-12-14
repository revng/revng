//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/AdjustStackPointer/AdjustStackPointerPass.h"

using namespace llvm;

bool adjustStackPointer(Function &F) {

  auto *StackInitFun = F.getParent()->getFunction("revng_init_local_sp");
  if (nullptr == StackInitFun)
    return false;

  CallInst *CallStackInit = nullptr;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *Call = dyn_cast<CallInst>(&I);
      // Skip instructions that are not CallInst and calls to Functions
      // different from revng_init_local_sp
      if (nullptr == Call or Call->getCalledFunction() != StackInitFun)
        continue;
      revng_assert(nullptr == CallStackInit);
      CallStackInit = Call;
    }
  }

  if (nullptr == CallStackInit)
    return false;

  auto *FType = StackInitFun->getFunctionType();
  revng_assert(FType->getNumParams() == 1U);
  auto *OffsetType = llvm::cast<IntegerType>(FType->getParamType(0U));
  unsigned NBits = OffsetType->getBitWidth();
  APInt AbsOfLowerNegativeOffset(NBits /* bit width */,
                                 0ULL /* initial value*/);

  for (Use &U : CallStackInit->uses()) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(U.getUser())) {

      // Only consider add and sub instructions
      auto OpKind = BinOp->getOpcode();
      if (OpKind != Instruction::Add and OpKind != Instruction::Sub)
        continue;

      // Get the other operand (the offset that is being added or subtracted
      // from the stack pointer.
      auto SPOperandIndex = U.getOperandNo();
      auto OtherOpIndex = SPOperandIndex ? 0U : 1U;
      auto *OtherOperand = U.getUser()->getOperand(OtherOpIndex);

      // In sub instructions, the stack pointer must be the first operand.
      // If we find a sub instruction where the stack pointer is subtracted from
      // another value, then the result is never a pointer and we have to bail
      // out.
      if (SPOperandIndex != 0 and OpKind == Instruction::Sub)
        continue;

      // Ignore cases where the offset is not constant.
      auto *ConstOffset = dyn_cast<ConstantInt>(OtherOperand);
      if (nullptr == ConstOffset)
        continue;

      auto Offset = ConstOffset->getValue();

      // Ensure that the Offset is always in the range of values that can be
      // represented exactly with NBits bits.
      // The minimum signed value is not valid, because it is not possible to
      // represent its negation exactly with NBits bits.
      auto SMax = APInt::getSignedMaxValue(NBits).getSExtValue();
      auto SMin = APInt::getSignedMinValue(NBits).getSExtValue();
      revng_assert(Offset.getSExtValue() > SMin
                   and Offset.getSExtValue() <= SMax);
      // Make both Offset and LowerNegativeOffset have the same number of bits.
      Offset = Offset.sextOrTrunc(NBits);

      if (OpKind == Instruction::Add) {
        // Only if Offset is strictly less than zero we might need to update
        // AbsOfLowerNegativeOffset.
        if (Offset.isNegative()) {
          // If Offset + AbsOfLowerNegativeOffset is also strictly less than
          // zero, then Offset is the lower negative offset we have found so
          // far, so we need to update AbsOfLowerNegativeOffset.
          bool Overflow = false;
          APInt Diff = Offset.sadd_ov(AbsOfLowerNegativeOffset, Overflow);
          revng_assert(not Overflow);
          if (Diff.isNegative()) {
            // Subtract the Diff from AbsOfLowerNegativeOffset.
            // Here Diff is always negative, so subtracting it from
            // AbsOfLowerNegativeOffset always yields a strictly positive value.
            // It is expected to never overflow, because of the assumptions we
            // have made on Offset.
            APInt Tmp = AbsOfLowerNegativeOffset.ssub_ov(Diff, Overflow);
            revng_assert(not Overflow);
            AbsOfLowerNegativeOffset = Tmp;
          }
        }
      } else /* (OpKind == Instruction::Sub) */ {
        // If Offset is larger than AbsOfLowerNegativeOffset, Offset is the
        // larger value that we have subtracted from the stack pointer so far,
        // so we have to update AbsOfLowerNegativeOffset.
        if (Offset.isStrictlyPositive()
            and Offset.sgt(AbsOfLowerNegativeOffset)) {
          AbsOfLowerNegativeOffset = Offset;
        }
      }
    }
  }
  revng_assert(AbsOfLowerNegativeOffset.isNonNegative());

  bool NeedsChange = AbsOfLowerNegativeOffset.isStrictlyPositive();
  if (NeedsChange) {
    // Start inserting instructions after the CallStackInit call.
    IRBuilder<> Builder(CallStackInit);

    // Create a new call to revng_init_local_sp, which takes the adjusted
    // offset as argument.
    Value *OffsetValue = ConstantInt::get(OffsetType, AbsOfLowerNegativeOffset);
    auto *CallInitAdjustedStack = Builder.CreateCall(StackInitFun, OffsetValue);

    // Zero extend the offset value if it's smaller than the return type of the
    // StackInitFun.
    auto *RetType = cast<IntegerType>(StackInitFun->getReturnType());
    revng_assert(RetType->getBitWidth() >= OffsetType->getBitWidth());
    if (RetType->getBitWidth() > OffsetType->getBitWidth())
      OffsetValue = Builder.CreateZExt(OffsetValue, RetType);

    // Create an addition that adjusts the stack poiner value returned from
    // CallInitAdjustedStack, by summing to it the offset value.
    // The idea behind this operation is that, later on, one can run a constant
    // propagation that will remove all the accesses to the stack pointer at
    // negative constant offsets. Nevertheless, the adjusted offset will remain
    // as argument of the new call to revng_init_local_sp. This operation is
    // necessary to ease the DLA, which currently does not support negative
    // offsets well.
    auto *AdjustedSP = Builder.CreateAdd(CallInitAdjustedStack, OffsetValue);
    CallStackInit->replaceAllUsesWith(AdjustedSP);

    // Now that it has no more uses, we can erase the previous call to
    // revng_init_local_sp, that is not necessary.
    CallStackInit->eraseFromParent();
  }

  revng_assert(not verifyFunction(F));
  return NeedsChange;
}

using ASPPass = LegacyPMAdjustStackPointerPass;

bool ASPPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
    return false;

  return adjustStackPointer(F);
}

void ASPPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadModelWrapperPass>();
}

// Registration code for the pass, using the legacy PassManager.
char ASPPass::ID = 0;

using RegPass = RegisterPass<ASPPass>;
static RegPass X("adjust-stack-pointer", "Adjust Stack Pointer Legacy Pass");
