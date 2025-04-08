/// \file ScopeGraphUtils.cpp
/// Helpers for the `ScopeGraph` building
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

#include "revng/RestructureCFG/ScopeGraphUtils.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

/// Helper function which converts `kebab-case` to `snake_case`, which we use to
/// convert the `FunctionTag` name to the actual name of the marker function
static std::string kebabToSnake(llvm::StringRef KebabString) {
  std::string Result(KebabString);
  std::replace(Result.begin(), Result.end(), '-', '_');
  return Result;
}

inline RegisterIRHelper
  ScopeCloserMarker(kebabToSnake(FunctionTags::ScopeCloserMarker.name().str()),
                    "");
inline RegisterIRHelper
  GotoBlockMarker(kebabToSnake(FunctionTags::GotoBlockMarker.name().str()), "");

// Helper function which set the attributes for the created function
// prototypes
static void setFunctionAttributes(Function *F, FunctionTags::Tag &Tag) {
  F->setLinkage(GlobalValue::ExternalLinkage);
  F->addFnAttr(Attribute::OptimizeNone);
  F->addFnAttr(Attribute::NoInline);
  F->addFnAttr(Attribute::NoMerge);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  F->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());

  // Add the custom tag to the `Function`
  Tag.addTo(F);

  // Add the `Marker` tag to the function, which signals that a function is a
  // marker for the `BasicBlock`/`TerminatorInst`
  FunctionTags::Marker.addTo(F);
}

static Function *getOrCreateScopeCloserFunction(Module *M) {
  FunctionTags::Tag &Tag = FunctionTags::ScopeCloserMarker;
  Function *Result = getUniqueFunctionWithTag(Tag, M);

  // Create the `ScopeCloserMarker` function if it doesn't exists
  if (not Result) {
    PointerType *BlockAddressTy = Type::getInt8PtrTy(getContext(M));
    auto *FT = FunctionType::get(Type::getVoidTy(getContext(M)),
                                 { BlockAddressTy },
                                 false);
    Result = cast<Function>(getOrInsertIRHelper(kebabToSnake(Tag.name()),
                                                *M,
                                                FT)
                              .getCallee());
    setFunctionAttributes(Result, Tag);
  }
  revng_assert(Result != nullptr);
  return Result;
}

static Function *getOrCreateGotoBlockFunction(Module *M) {
  FunctionTags::Tag &Tag = FunctionTags::GotoBlockMarker;
  Function *Result = getUniqueFunctionWithTag(Tag, M);

  // Create the `GotoBlockMarker` function if it doesn't exists
  if (not Result) {
    auto *FT = FunctionType::get(Type::getVoidTy(getContext(M)), {}, false);
    Result = cast<Function>(getOrInsertIRHelper(kebabToSnake(Tag.name()),
                                                *M,
                                                FT)
                              .getCallee());
    setFunctionAttributes(Result, Tag);
  }
  revng_assert(Result != nullptr);
  return Result;
}

ScopeGraphBuilder::ScopeGraphBuilder(Function *F) :
  ScopeCloserFunction(getOrCreateScopeCloserFunction(F->getParent())),
  GotoBlockFunction(getOrCreateGotoBlockFunction(F->getParent())) {
}

void ScopeGraphBuilder::makeGoto(BasicBlock *GotoBlock) {
  // We must have a `GotoBlock`
  revng_assert(GotoBlock);

  // We assume that when inserting a `goto` edge, the original block had a
  // single regular successor on the CFG
  Instruction *Terminator = GotoBlock->getTerminator();
  revng_assert(Terminator->getNumSuccessors() == 1);

  // We always insert the marker as the penultimate instruction in a
  // `BasicBlock`
  IRBuilder<> Builder(Terminator);
  Builder.CreateCall(GotoBlockFunction, {});
}

void ScopeGraphBuilder::addScopeCloser(BasicBlock *Source, BasicBlock *Target) {
  // We must have an insertion point
  revng_assert(Source);

  // We always insert the marker as the penultimate instruction in a
  // `BasicBlock`
  Instruction *Terminator = Source->getTerminator();
  IRBuilder<> Builder(Terminator);
  auto *BasicBlockAddressTarget = BlockAddress::get(Target);
  revng_assert(BasicBlockAddressTarget);
  Builder.CreateCall(ScopeCloserFunction, BasicBlockAddressTarget);
}

BasicBlock *ScopeGraphBuilder::makeGotoEdge(BasicBlock *Source,
                                            BasicBlock *Target) {
  Function *F = Source->getParent();

  // Create the `goto` block, and connect it with the `Target`
  LLVMContext &Context = getContext(Source);
  BasicBlock *GotoBlock = BasicBlock::Create(Context,
                                             "goto_" + Target->getName().str(),
                                             F);
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(GotoBlock);
  Builder.CreateBr(Target);

  // Insert the `goto_block` marker in the `ScopeGraph`
  ScopeGraphBuilder SGBuilder(F);
  SGBuilder.makeGoto(GotoBlock);

  // Redirect all the edges `Source` -> `Target` to `Source` -> `GotoBlock`
  auto SourceTerminator = Source->getTerminator();
  SourceTerminator->replaceSuccessorWith(Target, GotoBlock);

  return GotoBlock;
}

SmallVector<const Instruction *, 2>
getLast2InstructionsBeforeTerminator(const BasicBlock *BB) {
  SmallVector<const Instruction *, 2> Result;
  for (const auto &Group : enumerate(reverse(*BB))) {
    if (Group.index() > 2) {
      return Result;
    }
    const Instruction &I = Group.value();
    if (not I.isTerminator())
      Result.push_back(&I);
  }
  return Result;
}

BasicBlock *getScopeCloserTarget(const BasicBlock *BB) {

  // We must be provided with a `BasicBlock` where to search for the marker
  revng_assert(BB);

  // We search for the `scope_closer` marker in the last but one and last buttwo
  // instructions in `BB`
  for (const Instruction *I : getLast2InstructionsBeforeTerminator(BB)) {
    if (const CallInst
          *MarkerCall = getCallToTagged(I, FunctionTags::ScopeCloserMarker)) {
      auto *Operand = MarkerCall->getArgOperand(0);
      auto *TargetBlockAddress = cast<BlockAddress>(Operand);
      auto *TargetBB = TargetBlockAddress->getBasicBlock();
      return TargetBB;
    }
  }

  return nullptr;
}

bool isGotoBlock(const BasicBlock *BB) {

  // We must be provided with a `BasicBlock` where to search for the marker
  revng_assert(BB);

  // We search for the `scope_closer` marker in the last but one and last buttwo
  // instructions in `BB`
  for (const Instruction *I : getLast2InstructionsBeforeTerminator(BB)) {
    if (const CallInst
          *MarkerCall = getCallToTagged(I, FunctionTags::GotoBlockMarker)) {
      return true;
    }
  }

  return false;
}

void verifyScopeGraphAnnotationsImpl(FunctionTags::Tag &Tag,
                                     const BasicBlock *BB) {
  // We should find at maximum one occurrence of the call to the marker
  // function in either the last but one or last but two position in the
  // `BasicBlock`
  bool MarkerFound = false;
  for (const Instruction *I : getLast2InstructionsBeforeTerminator(BB)) {
    if (const CallInst *MarkerCall = getCallToTagged(I, Tag)) {
      revng_assert(MarkerFound == false, "Duplicate Marker Call");
      MarkerFound = true;
    }
  }

  // In the rest of the `BasicBlock`, we should not find any call to the marker
  // function
  for (const auto &Group : enumerate(reverse(*BB))) {
    if (Group.index() <= 2)
      continue;
    if (const CallInst *MarkerCall = getCallToTagged(&Group.value(), Tag)) {
      revng_abort("No ScopeGraph Marker Call expected in this portion of the "
                  "BasicBlock");
    }
  }

  // Additionally, the `ScopeGraph` has the requirement of permitting a single
  // successor for a `BasicBlock` which contains the `goto_block` marker
  if (MarkerFound and &Tag == &FunctionTags::GotoBlockMarker) {
    const Instruction *Terminator = BB->getTerminator();
    revng_assert(Terminator->getNumSuccessors() == 1);
  }
}

void verifyScopeGraphAnnotations(const BasicBlock *BB) {
  verifyScopeGraphAnnotationsImpl(FunctionTags::ScopeCloserMarker, BB);
  verifyScopeGraphAnnotationsImpl(FunctionTags::GotoBlockMarker, BB);
}
