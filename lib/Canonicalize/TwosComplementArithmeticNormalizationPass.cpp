//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

struct TwosComplementArithmeticNormalizationPass : public llvm::FunctionPass {
public:
  static char ID;

  TwosComplementArithmeticNormalizationPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

using TANP = TwosComplementArithmeticNormalizationPass;

class UnaryMinusBuilder {

  OpaqueFunctionsPool<llvm::Type *> Pool;
  llvm::IRBuilder<> Builder;

public:
  UnaryMinusBuilder(llvm::Function &F) :
    Pool(FunctionTags::UnaryMinus.getPool(*F.getParent())),
    Builder(F.getContext()) {}

  void SetInsertPoint(llvm::Instruction *I) { Builder.SetInsertPoint(I); }

  llvm::CallInst *operator()(llvm::Type *IntType, llvm::APInt Value) {
    revng_assert(llvm::isa<llvm::IntegerType>(IntType));
    llvm::Function *Func = Pool.get(IntType, IntType, IntType, "unary_minus");
    auto ConstInt = llvm::ConstantInt::getSigned(IntType,
                                                 Value.abs().getLimitedValue());
    return Builder.CreateCall(Func, { ConstInt });
  }
};

class BinaryNotBuilder {

  OpaqueFunctionsPool<llvm::Type *> Pool;
  llvm::IRBuilder<> Builder;

public:
  BinaryNotBuilder(llvm::Function &F) :
    Pool(FunctionTags::BinaryNot.getPool(*F.getParent())),
    Builder(F.getContext()) {}

  void SetInsertPoint(llvm::Instruction *I) { Builder.SetInsertPoint(I); }

  llvm::CallInst *operator()(llvm::Type *IntType, llvm::Value *Val) {
    revng_assert(isa<llvm::IntegerType>(IntType));
    llvm::Function *Func = Pool.get(IntType, IntType, IntType, "binary_not");
    return Builder.CreateCall(Func, { Val });
  }
};

class BooleanNotBuilder {

  OpaqueFunctionsPool<llvm::Type *> Pool;
  llvm::IRBuilder<> Builder;

public:
  BooleanNotBuilder(llvm::Function &F) :
    Pool(FunctionTags::BooleanNot.getPool(*F.getParent())),
    Builder(F.getContext()) {}

  void SetInsertPoint(llvm::Instruction *I) { Builder.SetInsertPoint(I); }

  llvm::CallInst *operator()(llvm::Type *IntType, llvm::Value *Val) {
    revng_assert(isa<llvm::IntegerType>(IntType));
    llvm::Function *Func = Pool.get(IntType,
                                    Builder.getIntNTy(1),
                                    IntType,
                                    "boolean_not");
    return Builder.CreateCall(Func, { Val });
  }
};

using Predicate = llvm::ICmpInst::Predicate;

bool TANP::runOnFunction(llvm::Function &F) {
  using namespace llvm;
  using namespace PatternMatch;

  UnaryMinusBuilder BuildUnaryMinus{ F };
  BinaryNotBuilder BuildBinaryNot{ F };
  BooleanNotBuilder BuildBooleanNot{ F };
  llvm::IRBuilder<> Builder{ F.getContext() };

  bool Changed = false;

  SmallVector<Instruction *, 8> DeadInsts;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {

      if (auto *CallToIsolated = getCallToIsolatedFunction(&I)) {

        for (Use &OperandUse : CallToIsolated->operands()) {

          auto *ConstantOperand = dyn_cast<ConstantInt>(OperandUse.get());
          if (not ConstantOperand)
            continue;

          if (ConstantOperand->isNegative()) {
            BuildUnaryMinus.SetInsertPoint(&I);
            auto UnaryMinus = BuildUnaryMinus(ConstantOperand->getType(),
                                              ConstantOperand->getValue());
            OperandUse.set(UnaryMinus);
          }
        }
        continue;
      }

      Value *NewV = nullptr;

      Value *Val = nullptr;
      const APInt *Int = nullptr;

      if ((match(&I, m_Xor(m_Value(Val), m_APInt(Int)))
           or match(&I, m_Xor(m_APInt(Int), m_Value(Val))))
          and Int->isAllOnesValue()) {
        BuildBinaryNot.SetInsertPoint(&I);
        NewV = BuildBinaryNot(I.getType(), Val);

      } else if (match(&I, m_Add(m_Value(Val), m_APInt(Int)))
                 and Int->isNegative()) {
        Builder.SetInsertPoint(&I);
        NewV = Builder.CreateSub(Val,
                                 ConstantInt::get(I.getType(), ~(*Int) + 1));
      } else if (match(&I, m_Sub(m_Value(Val), m_APInt(Int)))
                 and Int->isNegative()) {
        Builder.SetInsertPoint(&I);
        NewV = Builder.CreateAdd(Val,
                                 ConstantInt::get(I.getType(), ~(*Int) + 1));
      } else if ((match(&I, m_Mul(m_Value(Val), m_APInt(Int)))
                  or match(&I, m_Mul(m_APInt(Int), m_Value(Val))))
                 and Int->isNegative()) {
        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(Val->getType(), *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateMul(Val, UnaryMinus);
        }

      } else if (match(&I, m_SDiv(m_Value(Val), m_APInt(Int)))
                 and Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(Val->getType(), *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateSDiv(Val, UnaryMinus);
        }

      } else if (match(&I, m_SDiv(m_APInt(Int), m_Value(Val)))
                 and Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(Val->getType(), *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateSDiv(UnaryMinus, Val);
        }

      } else if (match(&I, m_SRem(m_Value(Val), m_APInt(Int)))
                 and Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(Val->getType(), *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateSRem(Val, UnaryMinus);
        }
      } else if (match(&I, m_SRem(m_APInt(Int), m_Value(Val)))
                 and Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(Val->getType(), *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateSRem(UnaryMinus, Val);
        }

      } else if (Predicate Pred;
                 match(&I, m_ICmp(Pred, m_Value(Val), m_APInt(Int)))) {
        const auto IntType = Val->getType();

        llvm::Value *X = nullptr;
        const APInt *RHS = nullptr;
        if (match(Val, m_Add(m_Value(X), m_APInt(RHS)))
            or match(Val, m_Sub(m_Value(X), m_APInt(RHS)))) {
          // Compute the new RHS if we move the RHS to the right of the
          // comparison operator, adjusting the old value of Int.
          using llvm::Instruction::Add;
          bool IsAdd = cast<llvm::Instruction>(Val)->getOpcode() == Add;
          APInt NewRHS = IsAdd ? (*Int - *RHS) : (*Int + *RHS);
          Builder.SetInsertPoint(I.getNextNonDebugInstruction());
          NewV = Builder.CreateICmp(Pred, X, ConstantInt::get(IntType, NewRHS));

          // If the predicate is relational, I is an inequality, meaning that it
          // has a range of results, that wraps around, and we have to take care
          // of that to avoid breaking semantics.
          if (llvm::ICmpInst::isRelational(Pred)) {
            unsigned BitWidth = RHS->getBitWidth();
            bool IsSigned = llvm::ICmpInst::isSigned(Pred);

            APInt Min = IsSigned ? APInt::getSignedMinValue(BitWidth) :
                                   /*Unsigned*/ APInt::getMinValue(BitWidth);
            APInt Max = IsSigned ? APInt::getSignedMaxValue(BitWidth) :
                                   /*Unsigned*/ APInt::getMaxValue(BitWidth);

            // Helper constant to discriminate between the two cases for the
            // solution of the inequalities.
            APInt MaxAddSubRHS = IsAdd ? (Max + *RHS) : (Max - *RHS);
            bool IntIntersectsAfterWrap = IsSigned ? Int->sle(MaxAddSubRHS) :
                                                     Int->ule(MaxAddSubRHS);

            // Compute the wrapping predicate, that always depends only on the
            // old predicate and its signedness.
            Predicate InversePred = llvm::ICmpInst::getInversePredicate(Pred);
            Predicate
              WrappingPred = llvm::ICmpInst::getStrictPredicate(InversePred);

            // Helper limits for computing the new constant operand of the
            // comparison after wrapping.
            APInt LastXBeforeWrap = IsAdd ? (Max - *RHS) : (Max + *RHS);
            APInt FirstXAfterWrap = IsAdd ? (Min - *RHS) : (Min + *RHS);

            // Here WrappingPred can only be GT or LT.
            // The WrapConstant we want to use for the new comparison only
            // depends on this.
            // Signedness may vary but it's not important because it's been
            // taken care of in our computation of WrappingPred.
            bool WrappedIsGreater = llvm::ICmpInst::isGT(WrappingPred);
            auto *WrapConstant = llvm::ConstantInt::get(IntType,
                                                        WrappedIsGreater ?
                                                          LastXBeforeWrap :
                                                          FirstXAfterWrap);

            llvm::Value *WrappingComparison = Builder.CreateICmp(WrappingPred,
                                                                 X,
                                                                 WrapConstant);

            if (IntIntersectsAfterWrap == WrappedIsGreater)
              NewV = Builder.CreateAnd(NewV, WrappingComparison);
            else
              NewV = Builder.CreateOr(NewV, WrappingComparison);
          }
        } else if (Int->isNegative()) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(IntType, *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateICmp(Pred, Val, UnaryMinus);
        } else if (Pred == Predicate::ICMP_EQ and Int->isNullValue()) {
          BuildBooleanNot.SetInsertPoint(&I);
          NewV = BuildBooleanNot(Val->getType(), Val);
        }
      }

      if (NewV) {
        Changed = true;
        I.replaceAllUsesWith(NewV);
        DeadInsts.emplace_back(&I);
      }
    }
  }

  for (auto *I : DeadInsts)
    llvm::RecursivelyDeleteTriviallyDeadInstructions(I);

  return Changed;
}

char TANP::ID = 0;

static llvm::RegisterPass<TANP> X("twoscomplement-normalization",
                                  "A simple pass that transforms arithmetic "
                                  "operations based on their two complements.",
                                  false,
                                  false);
