#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/Support/Debug.h"

namespace revng {

class IRBuilder;

namespace detail {

// NOLINTNEXTLINE
class RevngIRInsertionChecker : public llvm::IRBuilderDefaultInserter {
private:
  const revng::IRBuilder *Builder;

public:
  RevngIRInsertionChecker(const revng::IRBuilder &Builder) :
    Builder(&Builder) {}

  virtual void
  InsertHelper(llvm::Instruction *I,
               const llvm::Twine &Name,
               llvm::BasicBlock *BB,
               llvm::BasicBlock::iterator InsertPt) const override {
    checkImpl(); // < this dance is performed so we can inject these checks here

    // NOLINTNEXTLINE
    return llvm::IRBuilderDefaultInserter::InsertHelper(I, Name, BB, InsertPt);
  }

private:
  void checkImpl() const;
};

// NOLINTNEXTLINE
using LLVMBuilderBase = llvm::IRBuilder<llvm::ConstantFolder,
                                        detail::RevngIRInsertionChecker>;

} // namespace detail

/// This is a wrapper over llvm's IR builder that force-sets a debug location
/// even when its insertion point is a basic block.
///
/// On top of that, it asserts that *every* created instruction is created
/// with valid debug information attached.
class IRBuilder : public detail::LLVMBuilderBase {
public:
  void SetInsertPoint(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(BB);

    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  void SetInsertPoint(llvm::BasicBlock *BB) {
    SetInsertPoint(BB,
                   BB->getTerminator() ? BB->getTerminator()->getDebugLoc() :
                                         llvm::DebugLoc{});
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
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::LLVMContext &C) :
    detail::LLVMBuilderBase(BB->getContext(),
                            llvm::ConstantFolder{},
                            detail::RevngIRInsertionChecker{ *this }) {}

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(BB->getContext()) {

    SetInsertPoint(BB, DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::BasicBlock *BB) : IRBuilder(BB->getContext()) {
    SetInsertPoint(BB);
  }

  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::Instruction *I) : IRBuilder(BB->getContext()) {
    SetInsertPoint(I);
  }

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) :
    // NOLINTNEXTLINE
    IRBuilder(BB->getContext()) {
    SetInsertPoint(BB, I);
  }
};

inline void detail::RevngIRInsertionChecker::checkImpl() const {
  // TODO: adopt `isDebugLocationInvalid` here once it's available without
  //       the pipeline dependency.
  llvm::DebugLoc CurrentLocation = Builder->getCurrentDebugLocation();
  revng_assert(CurrentLocation);
  revng_assert(CurrentLocation->getScope());
  revng_assert(not CurrentLocation->getScope()->getName().empty());
}

} // namespace revng
