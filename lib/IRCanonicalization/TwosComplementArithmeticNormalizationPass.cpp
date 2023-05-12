//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/FunctionTags.h"

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
    Pool(F.getParent(), false), Builder(F.getContext()) {
    initUnaryMinusPool(Pool);
  }

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
    Pool(F.getParent(), false), Builder(F.getContext()) {
    initBinaryNotPool(Pool);
  }

  void SetInsertPoint(llvm::Instruction *I) { Builder.SetInsertPoint(I); }

  llvm::CallInst *operator()(llvm::Type *IntType, llvm::Value *Val) {
    revng_assert(isa<llvm::IntegerType>(IntType));
    llvm::Function *Func = Pool.get(IntType, IntType, IntType, "binary_not");
    return Builder.CreateCall(Func, { Val });
  }
};

static bool isSignedComparison(llvm::ICmpInst::Predicate P) {
  return P == llvm::ICmpInst::Predicate::ICMP_SGE
         or P == llvm::ICmpInst::Predicate::ICMP_SGT
         or P == llvm::ICmpInst::Predicate::ICMP_SLE
         or P == llvm::ICmpInst::Predicate::ICMP_SLT;
}

static bool isEqualityComparison(llvm::ICmpInst::Predicate P) {
  return P == llvm::ICmpInst::Predicate::ICMP_EQ
         or P == llvm::ICmpInst::Predicate::ICMP_NE;
}

bool TANP::runOnFunction(llvm::Function &F) {
  using namespace llvm;
  using namespace PatternMatch;

  UnaryMinusBuilder BuildUnaryMinus{ F };
  BinaryNotBuilder BuildBinaryNot{ F };
  llvm::IRBuilder<> Builder{ F.getContext() };

  bool Changed = false;

  SmallVector<Instruction *, 8> DeadInsts;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Value *Val = nullptr;
      const APInt *Int = nullptr;

      Value *NewV = nullptr;
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

      } else if (ICmpInst::Predicate Pred;
                 match(&I, m_ICmp(Pred, m_Value(Val), m_APInt(Int)))) {

        const auto IntType = Val->getType();

        llvm::Value *LHS = nullptr;
        const APInt *RHS = nullptr;
        if (match(Val, m_Add(m_Value(LHS), m_APInt(RHS)))) {
          bool Overflow = false;
          APInt NewInt = isSignedComparison(Pred) ?
                           Int->ssub_ov(*RHS, Overflow) :
                           Int->usub_ov(*RHS, Overflow);
          Builder.SetInsertPoint(I.getNextNonDebugInstruction());
          NewV = Builder.CreateICmp(Pred,
                                    LHS,
                                    ConstantInt::get(IntType, NewInt));

          if (not isEqualityComparison(Pred) and Overflow) {
            // Here we don't have overflow, and it's not an equality comparison,
            // so we have to handle wraparound
            llvm::ICmpInst::Predicate P;
            switch (Pred) {
            case llvm::ICmpInst::Predicate::ICMP_SGE:
            case llvm::ICmpInst::Predicate::ICMP_SGT: {
              P = llvm::ICmpInst::Predicate::ICMP_SLT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_UGE:
            case llvm::ICmpInst::Predicate::ICMP_UGT: {
              P = llvm::ICmpInst::Predicate::ICMP_ULT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_SLE:
            case llvm::ICmpInst::Predicate::ICMP_SLT: {
              P = llvm::ICmpInst::Predicate::ICMP_SGT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_ULE:
            case llvm::ICmpInst::Predicate::ICMP_ULT: {
              P = llvm::ICmpInst::Predicate::ICMP_UGT;
            } break;

            default:
              revng_abort();
            }
            auto *NotRHS = ConstantInt::get(IntType, ~*RHS);
            NewV = Builder.CreateAnd(NewV, Builder.CreateICmp(P, LHS, NotRHS));
          }
        } else if (match(Val, m_Sub(m_Value(LHS), m_APInt(RHS)))) {
          bool Overflow = false;
          APInt NewInt = isSignedComparison(Pred) ?
                           Int->ssub_ov(-*RHS, Overflow) :
                           Int->usub_ov(-*RHS, Overflow);
          Builder.SetInsertPoint(I.getNextNonDebugInstruction());
          NewV = Builder.CreateICmp(Pred,
                                    LHS,
                                    ConstantInt::get(IntType, NewInt));

          if (not isEqualityComparison(Pred) and Overflow) {
            // Here we don't have overflow, and it's not an equality comparison,
            // so we have to handle wraparound
            llvm::ICmpInst::Predicate P;
            switch (Pred) {
            case llvm::ICmpInst::Predicate::ICMP_SGE:
            case llvm::ICmpInst::Predicate::ICMP_SGT: {
              P = llvm::ICmpInst::Predicate::ICMP_SLT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_UGE:
            case llvm::ICmpInst::Predicate::ICMP_UGT: {
              P = llvm::ICmpInst::Predicate::ICMP_ULT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_SLE:
            case llvm::ICmpInst::Predicate::ICMP_SLT: {
              P = llvm::ICmpInst::Predicate::ICMP_SGT;
            } break;

            case llvm::ICmpInst::Predicate::ICMP_ULE:
            case llvm::ICmpInst::Predicate::ICMP_ULT: {
              P = llvm::ICmpInst::Predicate::ICMP_UGT;
            } break;

            default:
              revng_abort();
            }
            auto *NotMinusRHS = ConstantInt::get(IntType, ~-*RHS);
            NewV = Builder.CreateAnd(NewV,
                                     Builder.CreateICmp(P, LHS, NotMinusRHS));
          }
        } else if (Int->isSignBitSet()
                   and Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          BuildUnaryMinus.SetInsertPoint(&I);
          auto UnaryMinus = BuildUnaryMinus(IntType, *Int);
          Builder.SetInsertPoint(UnaryMinus->getNextNonDebugInstruction());
          NewV = Builder.CreateICmp(Pred, Val, UnaryMinus);
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
