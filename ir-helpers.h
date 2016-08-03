#ifndef _IRHELPERS_H
#define _IRHELPERS_H

// Standard includes
#include <set>

// LLVM includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/Interval.h"
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

/// Helper function to destroy an unconditional branch and, in case, the target
/// basic block, if it doesn't have any predecessors left.
static inline void purgeBranch(llvm::BasicBlock::iterator I) {
  auto *DeadBranch = llvm::dyn_cast<llvm::BranchInst>(I);
  // We allow only a branch and nothing else
  assert(DeadBranch != nullptr &&
         ++I == DeadBranch->getParent()->end());

  std::set<llvm::BasicBlock *> Successors;
  for (unsigned C = 0; C < DeadBranch->getNumSuccessors(); C++)
    Successors.insert(DeadBranch->getSuccessor(C));

  // Destroy the dead branch
  DeadBranch->eraseFromParent();

  // Check if someone else was jumping there and then destroy
  for (llvm::BasicBlock *BB : Successors)
    if (llvm::pred_empty(BB))
      BB->eraseFromParent();
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

static inline uint64_t getLimitedValue(llvm::Value *V) {
  return llvm::cast<llvm::ConstantInt>(V)->getLimitedValue();
}

static inline llvm::iterator_range<llvm::Interval::pred_iterator>
predecessors(llvm::Interval *BB) {
  return make_range(pred_begin(BB), pred_end(BB));
}

static inline llvm::iterator_range<llvm::Interval::succ_iterator>
successors(llvm::Interval *BB) {
  return make_range(succ_begin(BB), succ_end(BB));
}

template<typename T, unsigned I>
static inline void findOperand(llvm::Value *Op, T &Result) {
  abort();
}

template<typename T, unsigned I, typename Head, typename... Tail>
static inline void findOperand(llvm::Value *Op, T &Result) {
  using VT = typename std::remove_pointer<Head>::type;
  if (auto *Casted = llvm::dyn_cast<VT>(Op))
    std::get<I>(Result) = Casted;
  else
    return findOperand<T, I + 1, Tail...>(Op, Result);
}

/// \brief Returns a tuple of \p V's operands of the requested types
template<typename... T>
static inline std::tuple<T...> operandsByType(llvm::User *V) {
  std::tuple<T...> Result;
  unsigned OpCount = V->getNumOperands();
  assert(OpCount == sizeof...(T));

  for (llvm::Value *Op : V->operands())
    findOperand<std::tuple<T...>, 0, T...>(Op, Result);

  return Result;
}

/// \brief Return an range iterating backward from the given instruction
static inline llvm::iterator_range<llvm::BasicBlock::reverse_iterator>
backward_range(llvm::Instruction *I) {
  return llvm::make_range(llvm::make_reverse_iterator(I->getIterator()),
                          I->getParent()->rend());
}

#endif // _IRHELPERS_H
