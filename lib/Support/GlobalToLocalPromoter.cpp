//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"

#include "revng/Support/Assert.h"
#include "revng/Support/GlobalToLocalPromoter.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"

using Promoter = GlobalToLocalPromoter;

/// Carries the state of one indexing walk
class GlobalToLocalPromoter::Initializer {
private:
  using ConstantSet = llvm::DenseSet<const llvm::Constant *>;
  using MetadataSet = llvm::DenseSet<const llvm::Metadata *>;

private:
  Promoter &ThePromoter;
  const Filter &ShouldPromote;

  /// The globals the filter has turned down
  llvm::DenseSet<const llvm::GlobalVariable *> Rejected;

  /// The constant expressions that turned out not to refer to a selected
  /// global
  ConstantSet InspectedConstants;

  /// The metadata nodes that turned out not to name a selected global
  ///
  /// A node goes in as it is reached rather than once its own operands are
  /// done, which is sound only because finding a global in there is fatal.
  MetadataSet InspectedNodes;

public:
  Initializer(Promoter &ThePromoter, const Filter &ShouldPromote) :
    ThePromoter(ThePromoter), ShouldPromote(ShouldPromote) {}

public:
  /// Index every use of a selected global in \p F, in alphabetical order
  void run(llvm::Function &F);

private:
  void indexOperand(llvm::Use &TheUse);
  bool isSelected(const llvm::GlobalVariable &Global);
  void record(llvm::GlobalVariable *Global, llvm::Use &TheUse);

  /// \return the selected global \p Root is a use of, if any
  llvm::GlobalVariable *getUsedGlobal(llvm::Constant *Root);

  /// \return true if \p Root refers to a selected global
  bool refersToSelectedGlobal(const llvm::Constant *Root);

  /// \return true if \p Root names a selected global
  bool namesSelectedGlobal(const llvm::Metadata *Root);

  void sortByName();
};

void Promoter::Initializer::run(llvm::Function &F) {
  for (llvm::BasicBlock &BB : F)
    for (llvm::Instruction &I : BB)
      for (llvm::Use &TheUse : I.operands())
        indexOperand(TheUse);

  sortByName();
}

void Promoter::Initializer::indexOperand(llvm::Use &TheUse) {
  llvm::Value *Operand = TheUse.get();

  // Metadata names a value too, and rewriting such a use would take wrapping
  // the replacement in turn.
  if (auto *Wrapper = llvm::dyn_cast<llvm::MetadataAsValue>(Operand)) {
    if (namesSelectedGlobal(Wrapper->getMetadata()))
      revng_abort("A global variable to replace is named by metadata");

    return;
  }

  // Anything that is not a constant involves a global only through the
  // operands of the instruction that produced it, which the walk visits in
  // their own right.
  auto *AsConstant = llvm::dyn_cast<llvm::Constant>(Operand);
  if (AsConstant == nullptr)
    return;

  if (auto *Global = getUsedGlobal(AsConstant))
    record(Global, TheUse);
}

bool Promoter::Initializer::isSelected(const llvm::GlobalVariable &Global) {
  if (ThePromoter.Position.count(&Global) != 0)
    return true;

  if (Rejected.count(&Global) != 0)
    return false;

  if (not ShouldPromote or ShouldPromote(Global)) {
    revng_assert(Global.hasName());
    return true;
  }

  Rejected.insert(&Global);
  return false;
}

void Promoter::Initializer::record(llvm::GlobalVariable *Global,
                                   llvm::Use &TheUse) {
  auto &Index = ThePromoter.Index;
  auto &Position = ThePromoter.Position;

  auto [It, Inserted] = Position.try_emplace(Global, Index.size());
  if (Inserted)
    Index.push_back(IndexedGlobal{ .Global = Global });

  Index[It->second].Uses.push_back(&TheUse);
}

llvm::GlobalVariable *
Promoter::Initializer::getUsedGlobal(llvm::Constant *Root) {
  // Casts are the only expressions that can be rebuilt out of an alloca.
  auto *Stripped = llvm::cast<llvm::Constant>(skipCasts(Root));

  if (auto *Global = llvm::dyn_cast<llvm::GlobalVariable>(Stripped))
    return isSelected(*Global) ? Global : nullptr;

  // Constant data has no operands to look into, and is by far the common case.
  if (Stripped->getNumOperands() == 0)
    return nullptr;

  if (not InspectedConstants.insert(Stripped).second)
    return nullptr;

  if (refersToSelectedGlobal(Stripped)) {
    revng_abort("A global variable to replace is used by a constant "
                "expression that is not a cast");
  }

  return nullptr;
}

bool Promoter::Initializer::refersToSelectedGlobal(const llvm::Constant *Root) {
  llvm::SmallPtrSet<const llvm::Constant *, 8> Visited;
  llvm::SmallVector<const llvm::Constant *, 8> Queue = { Root };

  while (not Queue.empty()) {
    const llvm::Constant *Current = Queue.pop_back_val();
    if (not Visited.insert(Current).second)
      continue;

    // The initializer of a global is not a use within the function.
    if (auto *Global = llvm::dyn_cast<llvm::GlobalVariable>(Current)) {
      if (isSelected(*Global))
        return true;
      continue;
    }

    for (const llvm::Use &Operand : Current->operands())
      if (auto *Nested = llvm::dyn_cast<llvm::Constant>(Operand.get()))
        Queue.push_back(Nested);
  }

  return false;
}

bool Promoter::Initializer::namesSelectedGlobal(const llvm::Metadata *Root) {
  llvm::SmallVector<const llvm::Metadata *, 8> Queue = { Root };

  while (not Queue.empty()) {
    const llvm::Metadata *Current = Queue.pop_back_val();
    if (not InspectedNodes.insert(Current).second)
      continue;

    // A value is named either by a constant, which can refer to a global, or
    // by something local to a function, which cannot.
    if (auto *AsValue = llvm::dyn_cast<llvm::ValueAsMetadata>(Current)) {
      auto *Named = llvm::dyn_cast<llvm::Constant>(AsValue->getValue());
      if (Named != nullptr and refersToSelectedGlobal(Named))
        return true;

      continue;
    }

    if (auto *Node = llvm::dyn_cast<llvm::MDNode>(Current))
      for (const llvm::MDOperand &Operand : Node->operands())
        if (Operand)
          Queue.push_back(Operand.get());
  }

  return false;
}

void Promoter::Initializer::sortByName() {
  auto &Index = ThePromoter.Index;
  auto &Position = ThePromoter.Position;

  // The walk runs in traversal order: this sort is the single place the
  // alphabetical order is established.
  llvm::sort(Index, [](const IndexedGlobal &LHS, const IndexedGlobal &RHS) {
    return CompareByName(LHS.Global, RHS.Global);
  });

  Position.clear();
  for (unsigned I = 0; I < Index.size(); ++I)
    Position[Index[I].Global] = I;
}

Promoter::GlobalToLocalPromoter(const Filter &ShouldPromote,
                                llvm::Function &F) :
  TheFunction(&F) {
  Initializer TheInitializer(*this, ShouldPromote);
  TheInitializer.run(F);
}

Promoter::IndexedGlobal *Promoter::find(const llvm::GlobalVariable *Global) {
  auto It = Position.find(Global);
  if (It == Position.end())
    return nullptr;

  return &Index[It->second];
}

bool Promoter::replaceWithAlloca(llvm::GlobalVariable *Global,
                                 llvm::AllocaInst *Alloca) {
  revng_assert(Global != nullptr);
  revng_assert(Alloca != nullptr);
  revng_assert(Alloca->getFunction() == TheFunction);

  IndexedGlobal *Indexed = find(Global);
  if (Indexed == nullptr)
    return false;

  revng_assert(not Indexed->Replaced,
               "A global variable can only be replaced once");
  Indexed->Replaced = true;

  revng::IRBuilder Builder(Alloca->getContext());

  for (llvm::Use *TheUse : Indexed->Uses) {
    llvm::Value *Operand = TheUse->get();

    if (Operand == Global) {
      TheUse->set(Alloca);
      continue;
    }

    auto *User = llvm::cast<llvm::Instruction>(TheUse->getUser());
    llvm::Instruction *InsertBefore = User;
    if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(User)) {
      // A `phi` has no instruction to insert before: the cast belongs at the
      // end of the block the value comes from.
      unsigned OperandNo = TheUse->getOperandNo();
      unsigned
        Incoming = llvm::PHINode::getIncomingValueNumForOperand(OperandNo);
      InsertBefore = Phi->getIncomingBlock(Incoming)->getTerminator();
    }

    Builder.SetInsertPoint(InsertBefore, InsertBefore->getDebugLoc());
    TheUse->set(Builder.CreateBitOrPointerCast(Alloca, Operand->getType()));
  }

  return not Indexed->Uses.empty();
}
