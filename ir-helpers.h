#ifndef _IRHELPERS_H
#define _IRHELPERS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

static inline void replaceInstruction(llvm::Instruction *Old,
                                      llvm::Instruction *New) {
  Old->replaceAllUsesWith(New);

  llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 2> Metadata;
  Old->getAllMetadata(Metadata);
  for (auto& MDPair : Metadata)
    New->setMetadata(MDPair.first, MDPair.second);

  Old->eraseFromParent();
}

/// Helper function to destroy an unconditional branch and, in case, the
/// target basic block, if it doesn't have any predecessors left.
static inline void purgeBranch(llvm::BasicBlock::iterator I) {
  auto *DeadBranch = llvm::dyn_cast<llvm::BranchInst>(I);
  // We allow only an unconditional branch and nothing else
  assert(DeadBranch != nullptr &&
         DeadBranch->isUnconditional() &&
         ++I == DeadBranch->getParent()->end());

  // Obtain the target of the dead branch
  llvm::BasicBlock *DeadBranchTarget = DeadBranch->getSuccessor(0);

  // Destroy the dead branch
  DeadBranch->eraseFromParent();

  // Check if someone else was jumping there and then destroy
  if (llvm::pred_empty(DeadBranchTarget))
    DeadBranchTarget->eraseFromParent();
}

static inline llvm::ConstantInt *getConstValue(llvm::Constant *C,
                                               const llvm::DataLayout &DL) {
  while (auto *Expr = llvm::dyn_cast<llvm::ConstantExpr>(C)) {
    C = ConstantFoldConstantExpression(Expr, DL);

    if (Expr->getOpcode() == llvm::Instruction::IntToPtr
        || Expr->getOpcode() == llvm::Instruction::PtrToInt)
      C = Expr->getOperand(0);
  }

  if (llvm::isa<llvm::ConstantPointerNull>(C)) {
    auto *Ptr = llvm::IntegerType::get(C->getType()->getContext(),
                                       DL.getPointerSizeInBits());
    return llvm::ConstantInt::get(Ptr, 0);
  }

  auto *Integer = llvm::cast<llvm::ConstantInt>(C);
  return Integer;
}

static inline uint64_t getSExtValue(llvm::Constant *C,
                                    const llvm::DataLayout &DL){
  return getConstValue(C, DL)->getSExtValue();
}

static inline uint64_t getZExtValue(llvm::Constant *C,
                                    const llvm::DataLayout &DL){
  return getConstValue(C, DL)->getZExtValue();
}

static inline uint64_t getExtValue(llvm::Constant *C,
                                   bool Sign,
                                   const llvm::DataLayout &DL){
  if (Sign)
    return getSExtValue(C, DL);
  else
    return getZExtValue(C, DL);
}

#endif // _IRHELPERS_H
