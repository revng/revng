#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

namespace revng {

namespace detail {

// NOLINTNEXTLINE
using LLVMBuilderBase = llvm::IRBuilder<>;

} // namespace detail

inline bool
sameSize(llvm::Type *LHS, llvm::Type *RHS, const llvm::DataLayout &DL) {
  auto LHSSize = DL.getTypeStoreSize(LHS).getFixedValue();
  auto RHSSize = DL.getTypeStoreSize(RHS).getFixedValue();
  return LHSSize == RHSSize;
}

/// This is a wrapper over llvm's IR builder that force-sets a debug location
/// even when its insertion point is a basic block.
class IRBuilder : public detail::LLVMBuilderBase {
public:
  //
  // These explicit `llvm::DebugLoc` overloads are revng-specific,
  // prefer them whenever applicable.
  //
  void SetInsertPoint(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(BB);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

  void SetInsertPoint(llvm::Instruction *I, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(I);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

  void SetInsertPoint(llvm::BasicBlock *BB,
                      llvm::BasicBlock::iterator I,
                      const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(BB, I);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

  void SetInsertPointPastAllocas(llvm::Function *F, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPointPastAllocas(F);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

public:
  //
  // These mirror the corresponding interfaces of llvm's IR builder,
  //
  void SetInsertPoint(llvm::BasicBlock *BB) {
    detail::LLVMBuilderBase::SetInsertPoint(BB);
    if (BB->getTerminator()) {
      auto DL = BB->getTerminator()->getDebugLoc();
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
    }
  }
  void SetInsertPoint(llvm::Instruction *I) {
    detail::LLVMBuilderBase::SetInsertPoint(I);
  }
  void SetInsertPoint(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) {
    detail::LLVMBuilderBase::SetInsertPoint(BB, I);
  }
  void SetInsertPointPastAllocas(llvm::Function *F) {
    detail::LLVMBuilderBase::SetInsertPointPastAllocas(F);
    auto DL = detail::LLVMBuilderBase::GetInsertPoint()->getDebugLoc();
    detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

public:
  /// Create an `alloca` of integer type or byte array capable to hold \p T
  llvm::AllocaInst *createSimpleAlloca(llvm::Type *T) {
    using namespace llvm;
    const DataLayout &DL = GetInsertBlock()->getModule()->getDataLayout();

    if (T->isPointerTy()) {
      auto Size = DL.getPointerSizeInBits(T->getPointerAddressSpace());
      return CreateAlloca(this->getIntNTy(Size));
    }

    if (T->isIntegerTy())
      return CreateAlloca(T);

    auto Size = DL.getTypeStoreSize(T).getFixedValue();
    return CreateAlloca(ArrayType::get(this->getInt8Ty(), Size));
  }

  /// Load a value of type \p DesiredType from \p Variable (an `AllocaInst` or
  /// `GlobalVariable`). The variable's allocated type and \p DesiredType must
  /// have the same store size.
  llvm::LoadInst *createLoadFromVariable(llvm::Value *Variable,
                                         llvm::Type *DesiredType) {
    using namespace llvm;
    Type *AllocatedType = getVariableType(Variable);
    const DataLayout &DL = GetInsertBlock()->getModule()->getDataLayout();
    revng_assert(sameSize(DesiredType, AllocatedType, DL));
    return this->CreateLoad(DesiredType, Variable);
  }

  /// Store \p V into \p Variable (an `AllocaInst` or `GlobalVariable`). The
  /// variable's allocated type and \p V's type must have the same store size.
  llvm::StoreInst *createStoreToVariable(llvm::Value *V,
                                         llvm::Value *Variable) {
    using namespace llvm;
    Type *AllocatedType = getVariableType(Variable);
    const DataLayout &DL = GetInsertBlock()->getModule()->getDataLayout();
    revng_assert(sameSize(V->getType(), AllocatedType, DL));
    return CreateStore(V, Variable);
  }

  llvm::LoadInst *createLoad(llvm::GlobalVariable *GV) {
    return this->CreateLoad(GV->getValueType(), GV);
  }

  llvm::LoadInst *createLoad(llvm::AllocaInst *Alloca) {
    return this->CreateLoad(Alloca->getAllocatedType(), Alloca);
  }

  llvm::LoadInst *createLoadVariable(llvm::Value *Variable) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(Variable))
      return createLoad(Alloca);
    if (auto *GV = llvm::dyn_cast<llvm::GlobalVariable>(Variable))
      return createLoad(GV);
    revng_abort("Either GlobalVariable or AllocaInst expected");
  }

  void setInsertPointToFirstNonAlloca(llvm::Function &F) {
    using namespace llvm;
    for (Instruction &I : F.getEntryBlock()) {
      if (not isa<AllocaInst>(&I)) {
        SetInsertPoint(&I);
        return;
      }
    }
    revng_abort();
  }

  llvm::SmallVector<llvm::Value *, 4> unpack(llvm::Value *V) {
    using namespace llvm;
    Type *T = V->getType();
    if (isa<IntegerType>(T))
      return { V };
    if (auto *ST = dyn_cast<StructType>(T)) {
      SmallVector<Value *, 4> Result;
      for (unsigned I = 0; I < ST->getNumElements(); ++I)
        Result.push_back(this->CreateExtractValue(V, { I }));
      return Result;
    }
    revng_abort("Cannot unpack the given type");
  }

public:
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::LLVMContext &C) : detail::LLVMBuilderBase(C) {}

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(BB->getContext()) {

    SetInsertPoint(BB, DL);
  }

  // NOLINTNEXTLINE
  IRBuilder(llvm::Instruction *I, const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(I->getContext()) {

    SetInsertPoint(I, DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::BasicBlock *BB) : IRBuilder(BB->getContext()) {
    SetInsertPoint(BB);
  }

  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::Instruction *I) : IRBuilder(I->getContext()) {
    SetInsertPoint(I);
  }

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) :
    // NOLINTNEXTLINE
    IRBuilder(BB->getContext()) {
    SetInsertPoint(BB, I);
  }
};

} // namespace revng
