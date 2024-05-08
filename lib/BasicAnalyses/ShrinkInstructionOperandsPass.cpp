/// \file ShrinkInstructionOperandsPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"

#include "revng/BasicAnalyses/ShrinkInstructionOperandsPass.h"
#include "revng/Support/IRHelpers.h"

enum Signedness {
  DontCare,
  Signed,
  Unsigned
};

static llvm::Value *getPreExt(llvm::Value *V, Signedness S) {
  if (auto *ZExt = llvm::dyn_cast<llvm::ZExtInst>(V)) {
    if (S != Signed)
      return ZExt->getOperand(0);
  } else if (auto *SExt = llvm::dyn_cast<llvm::SExtInst>(V)) {
    if (S != Unsigned)
      return SExt->getOperand(0);
  }

  return V;
}

static llvm::Instruction *getPostTrunc(llvm::Value *V) {
  if (llvm::User *U = getUniqueUser(V))
    return llvm::dyn_cast<llvm::TruncInst>(U);

  return nullptr;
}

static unsigned getSize(llvm::Value *V) {
  llvm::Type *T = V->getType();
  if (auto *IntTy = llvm::dyn_cast<llvm::IntegerType>(T))
    return IntTy->getBitWidth();

  return 0;
}

static void replaceAndResizeOperand(llvm::Instruction *I,
                                    unsigned Index,
                                    llvm::Value *V,
                                    unsigned NewSize,
                                    Signedness S) {
  if (getSize(I) == NewSize)
    return;

  llvm::IRBuilder<> Builder(I);
  llvm::Type *NewType = Builder.getIntNTy(NewSize);
  llvm::Value *NewOperand = nullptr;
  if (S == Signed) {
    NewOperand = Builder.CreateSExtOrTrunc(V, NewType);
  } else {
    NewOperand = Builder.CreateZExtOrTrunc(V, NewType);
  }

  I->setOperand(Index, NewOperand);
}

llvm::PreservedAnalyses
ShrinkInstructionOperandsPass::run(llvm::Function &F,
                                   llvm::FunctionAnalysisManager &FAM) {
  using namespace llvm;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {

      Value *ActualOp0 = nullptr;
      unsigned Actual0Size = 0;
      Value *ActualOp1 = nullptr;
      unsigned Actual1Size = 0;
      unsigned OriginalSize = getSize(&I);

      if (OriginalSize == 0)
        continue;

      switch (I.getOpcode()) {
      case Instruction::ICmp:

        if (auto *Compare = dyn_cast<ICmpInst>(&I)) {

          Signedness S;
          if (Compare->isSigned())
            S = Signed;
          else if (Compare->isUnsigned())
            S = Unsigned;
          else
            S = DontCare;

          Value *Op0 = I.getOperand(0);
          Value *Op1 = I.getOperand(1);
          revng_assert(getSize(Op0) == getSize(Op1));

          ActualOp0 = getPreExt(Op0, S);
          ActualOp1 = getPreExt(Op1, S);
          Actual0Size = getSize(ActualOp0);
          Actual1Size = getSize(ActualOp1);

          if (Actual0Size != 0 and Actual1Size != 0
              and Actual0Size != OriginalSize and Actual1Size != OriginalSize) {
            // OK, we can shrink
            unsigned NewSize = std::max(Actual0Size, Actual1Size);
            replaceAndResizeOperand(&I, 0, ActualOp0, NewSize, S);
            replaceAndResizeOperand(&I, 1, ActualOp1, NewSize, S);
          }
        }

        break;

      case Instruction::And:
      case Instruction::Or:
      case Instruction::Xor:
      case Instruction::Shl:
      case Instruction::Add:
      case Instruction::Mul:
      case Instruction::Sub: {
        Instruction *ActualOutput = getPostTrunc(&I);
        if (ActualOutput == nullptr)
          continue;

        unsigned OutputSize = getSize(ActualOutput);

        if (OutputSize != OriginalSize) {
          // OK, we can shrink
          ActualOp0 = getPreExt(I.getOperand(0), DontCare);
          ActualOp1 = getPreExt(I.getOperand(1), DontCare);
          replaceAndResizeOperand(&I, 0, ActualOp0, OutputSize, DontCare);
          replaceAndResizeOperand(&I, 1, ActualOp1, OutputSize, DontCare);
          I.mutateType(ActualOutput->getType());
          ActualOutput->replaceAllUsesWith(&I);
          eraseFromParent(ActualOutput);
        }

      } break;

      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::URem:
      case Instruction::SRem:
        // TODO
        break;

      case Instruction::AShr:
      case Instruction::LShr:
        // TODO
        break;

      default:
        break;
      }
    }
  }

  return PreservedAnalyses::none();
}
