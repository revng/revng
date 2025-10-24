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

class DebugInfoCheckingInserter : public llvm::IRBuilderDefaultInserter {
private:
  const revng::IRBuilder *const Builder = nullptr;
  const bool EnableChecks = true;

public:
  DebugInfoCheckingInserter(const revng::IRBuilder &Builder,
                            bool EnableChecks) :
    Builder(&Builder), EnableChecks(EnableChecks) {}

  virtual void
  InsertHelper(llvm::Instruction *I,
               const llvm::Twine &Name,
               llvm::BasicBlock *BB,
               llvm::BasicBlock::iterator InsertPt) const override {
    checkImpl(); // < this dance is performed so we can inject these checks here

    return llvm::IRBuilderDefaultInserter::InsertHelper(I, Name, BB, InsertPt);
  }

  void checkImpl() const;
};

// NOLINTNEXTLINE
using LLVMBuilderBase = llvm::IRBuilder<llvm::ConstantFolder,
                                        detail::DebugInfoCheckingInserter>;

} // namespace detail

/// This is a wrapper over llvm's IR builder that force-sets a debug location
/// even when its insertion point is a basic block.
///
/// On top of that, it asserts that *every* created instruction is created
/// with valid debug information attached.
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

    getInserter().checkImpl();
  }

  void SetInsertPoint(llvm::Instruction *I, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(I);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);

    getInserter().checkImpl();
  }

  void SetInsertPoint(llvm::BasicBlock *BB,
                      llvm::BasicBlock::iterator I,
                      const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPoint(BB, I);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);

    getInserter().checkImpl();
  }

  void SetInsertPointPastAllocas(llvm::Function *F, const llvm::DebugLoc &DL) {
    detail::LLVMBuilderBase::SetInsertPointPastAllocas(F);
    if (DL)
      detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);

    getInserter().checkImpl();
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

    getInserter().checkImpl();
  }
  void SetInsertPoint(llvm::Instruction *I) {
    detail::LLVMBuilderBase::SetInsertPoint(I);

    getInserter().checkImpl();
  }
  void SetInsertPoint(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) {
    detail::LLVMBuilderBase::SetInsertPoint(BB, I);

    getInserter().checkImpl();
  }
  void SetInsertPointPastAllocas(llvm::Function *F) {
    detail::LLVMBuilderBase::SetInsertPointPastAllocas(F);
    auto DL = detail::LLVMBuilderBase::GetInsertPoint()->getDebugLoc();
    detail::LLVMBuilderBase::SetCurrentDebugLocation(DL);

    getInserter().checkImpl();
  }

protected:
  // NOLINTNEXTLINE
  explicit IRBuilder(bool EnableDebugInformationChecks, llvm::LLVMContext &C) :
    detail::LLVMBuilderBase(C,
                            llvm::ConstantFolder{},
                            detail::DebugInfoCheckingInserter{
                              *this,
                              EnableDebugInformationChecks }) {}

public:
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::LLVMContext &C) : IRBuilder(true, C) {}

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(true, BB->getContext()) {

    SetInsertPoint(BB, DL);
  }

  // NOLINTNEXTLINE
  IRBuilder(llvm::Instruction *I, const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(true, I->getContext()) {

    SetInsertPoint(I, DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::BasicBlock *BB) : IRBuilder(true, BB->getContext()) {
    SetInsertPoint(BB);
  }

  // NOLINTNEXTLINE
  explicit IRBuilder(llvm::Instruction *I) : IRBuilder(true, I->getContext()) {
    SetInsertPoint(I);
  }

  // NOLINTNEXTLINE
  IRBuilder(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) :
    // NOLINTNEXTLINE
    IRBuilder(true, BB->getContext()) {
    SetInsertPoint(BB, I);
  }
};

/// The reason `NonDebugInfoCheckingIRBuilder` inherits from
/// `revng::IRBuilder` instead of just aliasing `llvm::IRBuilder<>` is to allow
/// passing it into functions accepting `revng::IRBuilder`.
// NOLINTNEXTLINE
class NonDebugInfoCheckingIRBuilder : public IRBuilder {
public:
  explicit NonDebugInfoCheckingIRBuilder(llvm::LLVMContext &C) :
    // NOLINTNEXTLINE
    IRBuilder(false, C) {}

  NonDebugInfoCheckingIRBuilder(llvm::BasicBlock *BB,
                                const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(false, BB->getContext()) {

    SetInsertPoint(BB, DL);
  }

  NonDebugInfoCheckingIRBuilder(llvm::Instruction *I,
                                const llvm::DebugLoc &DL) :
    // NOLINTNEXTLINE
    IRBuilder(false, I->getContext()) {

    SetInsertPoint(I, DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  explicit NonDebugInfoCheckingIRBuilder(llvm::BasicBlock *BB) :
    // NOLINTNEXTLINE
    IRBuilder(false, BB->getContext()) {
    SetInsertPoint(BB);
  }

  explicit NonDebugInfoCheckingIRBuilder(llvm::Instruction *I) :
    // NOLINTNEXTLINE
    IRBuilder(false, I->getContext()) {
    SetInsertPoint(I);
  }

  // NOLINTNEXTLINE
  NonDebugInfoCheckingIRBuilder(llvm::BasicBlock *BB,
                                llvm::BasicBlock::iterator I) :
    // NOLINTNEXTLINE
    IRBuilder(false, BB->getContext()) {
    SetInsertPoint(BB, I);
  }
};

inline void detail::DebugInfoCheckingInserter::checkImpl() const {
  if (EnableChecks) {
    // TODO: adopt `isDebugLocationInvalid` here once it's available without
    //       the pipeline dependency.
    llvm::DebugLoc CurrentLocation = Builder->getCurrentDebugLocation();
    revng_assert(CurrentLocation);
    revng_assert(CurrentLocation->getScope());
    revng_assert(not CurrentLocation->getScope()->getName().empty());
  }
}

} // namespace revng
