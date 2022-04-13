//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

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

  SmallVector<Instruction *, 8> DeadInsts;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {

      Value *LHS = nullptr;
      const APInt *RHS = nullptr;
      Value *NewV = nullptr;
      Builder.SetInsertPoint(&I);

      if (match(&I, m_Add(m_Value(LHS), m_APInt(RHS))) && RHS->isNegative()) {
        NewV = Builder.CreateSub(LHS,
                                 ConstantInt::get(I.getType(), ~(*RHS) + 1));
      } else if (match(&I, m_Sub(m_Value(LHS), m_APInt(RHS)))
                 && RHS->isNegative()) {
        NewV = Builder.CreateAdd(LHS,
                                 ConstantInt::get(I.getType(), ~(*RHS) + 1));
      } else if (match(&I, m_Mul(m_Value(LHS), m_APInt(RHS)))
                 && RHS->isNegative()) {
        NewV = Builder.CreateMul(LHS,
                                 ConstantInt::get(I.getType(), -(~(*RHS) + 1)));
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
