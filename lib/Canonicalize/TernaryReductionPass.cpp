//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/OpaqueFunctionsPool.h"

struct TernaryReductionPass : public llvm::FunctionPass {
public:
  static char ID;

  TernaryReductionPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &Function) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

class TernaryReductionImpl {
  // Here we should definitely use the builder that checks the debug info,
  // but since this going to go away soon, let it stay as is.
  revng::NonDebugInfoCheckingIRBuilder Builder;
  OpaqueFunctionsPool<llvm::Type *> BooleanNotPool;

public:
  TernaryReductionImpl(llvm::Module &Module) :
    Builder(Module.getContext()),
    BooleanNotPool(FunctionTags::BooleanNot.getPool(Module)) {}

  llvm::Value *reduce(llvm::SelectInst &Select) {
    std::optional TrueBranch = unwrapBoolConstant(Select.getTrueValue());
    std::optional FalseBranch = unwrapBoolConstant(Select.getFalseValue());
    revng_assert(!TrueBranch || !FalseBranch,
                 "Both are constant booleans? Why is this even a select?");

    Builder.SetInsertPoint(&Select);
    if (TrueBranch.has_value()) {
      if (TrueBranch.value()) {
        return Builder.CreateOr(Select.getCondition(), Select.getFalseValue());
      } else {
        return Builder.CreateAnd(booleanNot(Select.getCondition()),
                                 Select.getFalseValue());
      }
    } else if (FalseBranch.has_value()) {
      if (FalseBranch.value()) {
        return Builder.CreateOr(booleanNot(Select.getCondition()),
                                Select.getTrueValue());
      } else {
        return Builder.CreateAnd(Select.getCondition(), Select.getTrueValue());
      }
    }

    return nullptr;
  }

private:
  static std::optional<bool> unwrapBoolConstant(const llvm::Value *Value) {
    if (auto *Constant = llvm::dyn_cast<llvm::ConstantInt>(Value))
      if (Constant->getBitWidth() == 1)
        return !Constant->isZero();

    return std::nullopt;
  }

  llvm::Value *booleanNot(llvm::Value *Value) {
    auto *BooleanNotFunction = BooleanNotPool.get(Value->getType(),
                                                  Builder.getIntNTy(1),
                                                  Value->getType(),
                                                  "boolean_not");
    return Builder.CreateCall(BooleanNotFunction, { Value });
  }
};

bool TernaryReductionPass::runOnFunction(llvm::Function &Function) {
  TernaryReductionImpl Helper(*Function.getParent());
  llvm::SmallVector<llvm::WeakTrackingVH, 8> ToRemove;
  for (llvm::BasicBlock &BasicBlock : Function) {
    for (llvm::Instruction &Instruction : BasicBlock) {
      if (auto *Select = llvm::dyn_cast<llvm::SelectInst>(&Instruction)) {
        if (llvm::Value *Replacement = Helper.reduce(*Select)) {
          Instruction.replaceAllUsesWith(Replacement);
          ToRemove.emplace_back(&Instruction);
        }
      }
    }
  }

  RecursivelyDeleteTriviallyDeadInstructions(ToRemove);

  return !ToRemove.empty();
}

char TernaryReductionPass::ID = 0;

static llvm::RegisterPass<TernaryReductionPass> X("ternary-reduction",
                                                  "A pass for simplifying "
                                                  "ternary operation involving "
                                                  "constants.",
                                                  false,
                                                  false);
