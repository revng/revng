//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
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

bool TANP::runOnFunction(llvm::Function &F) {
  using namespace llvm;
  using namespace PatternMatch;

  IRBuilder<> Builder(F.getContext());

  OpaqueFunctionsPool<llvm::Type *> UnaryMinusPool(F.getParent(), false);
  initUnaryMinusPool(UnaryMinusPool);

  OpaqueFunctionsPool<llvm::Type *> BinaryNotPool(F.getParent(), false);
  initBinaryNotPool(BinaryNotPool);

  auto BuildUnaryMinus =
    [&Builder, &UnaryMinusPool](const Value *Val, const APInt *Int) -> Value * {
    const auto IntType = Val->getType();
    auto Func = UnaryMinusPool.get(IntType, IntType, IntType, "unary_minus");
    auto IntValue = Int->abs().getLimitedValue();
    auto Value = ConstantInt::getSigned(IntType, IntValue);
    auto Call = Builder.CreateCall(Func, { Value });
    return Call;
  };

  SmallVector<Instruction *, 8> DeadInsts;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Value *Val = nullptr;
      const APInt *Int = nullptr;
      Value *NewV = nullptr;
      ICmpInst::Predicate Pred;
      Builder.SetInsertPoint(&I);

      if (match(&I, m_Add(m_Value(Val), m_APInt(Int))) && Int->isNegative()) {
        NewV = Builder.CreateSub(Val,
                                 ConstantInt::get(I.getType(), ~(*Int) + 1));
      } else if (match(&I, m_Sub(m_Value(Val), m_APInt(Int)))
                 && Int->isNegative()) {
        NewV = Builder.CreateAdd(Val,
                                 ConstantInt::get(I.getType(), ~(*Int) + 1));
      } else if ((match(&I, m_Mul(m_Value(Val), m_APInt(Int)))
                  || match(&I, m_Mul(m_APInt(Int), m_Value(Val))))
                 && Int->isNegative()) {
        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateMul(Val, Call);
        }

      } else if (match(&I, m_SDiv(m_Value(Val), m_APInt(Int)))
                 && Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateSDiv(Val, Call);
        }

      } else if (match(&I, m_SDiv(m_APInt(Int), m_Value(Val)))
                 && Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateSDiv(Call, Val);
        }

      } else if (match(&I, m_SRem(m_Value(Val), m_APInt(Int)))
                 && Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateSRem(Val, Call);
        }
      } else if (match(&I, m_SRem(m_APInt(Int), m_Value(Val)))
                 && Int->isNegative()) {

        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateSRem(Call, Val);
        }
      } else if ((match(&I, m_Xor(m_Value(Val), m_APInt(Int)))
                  || match(&I, m_Xor(m_APInt(Int), m_Value(Val))))
                 && Int->isAllOnesValue()) {
        const auto IntType = I.getType();
        auto Func = BinaryNotPool.get(IntType, IntType, IntType, "binary_not");
        NewV = Builder.CreateCall(Func, { Val });
      } else if (match(&I, m_ICmp(Pred, m_Value(Val), m_APInt(Int)))
                 && Pred == ICmpInst::Predicate::ICMP_EQ) {
        const auto IntType = Val->getType();

        if (Int->isSignBitSet()
            && Int->isSignedIntN(IntType->getIntegerBitWidth())) {
          auto Call = BuildUnaryMinus(Val, Int);
          NewV = Builder.CreateICmp(Pred, Val, Call);
        }
      }

      if (NewV) {
        I.replaceAllUsesWith(NewV);
        DeadInsts.emplace_back(&I);
      }
    }
  }

  for (auto *I : DeadInsts)
    I->eraseFromParent();

  return not DeadInsts.empty();
}

char TANP::ID = 0;

static llvm::RegisterPass<TANP> X("twoscomplement-normalization",
                                  "A simple pass that transforms arithmetic "
                                  "operations based on their two complements.",
                                  false,
                                  false);
