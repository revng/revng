//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/MarkForSerialization/MarkForSerializationPass.h"

#include "Mangling.h"

struct AddSCEVBarrierPass : public llvm::FunctionPass {

  static char ID;

  AddSCEVBarrierPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.addUsedIfAvailable<MarkForSerializationPass>();
    AU.setPreservesAll();
  }
};

char AddSCEVBarrierPass::ID = 0;

using Pass = AddSCEVBarrierPass;
using llvm::RegisterPass;
static RegisterPass<Pass> X("add-scev-barrier", "Pass to add SCEV barriers");

static std::string makeTypeName(const llvm::Type *Ty) {
  std::string Name;
  if (auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(Ty)) {
    Name = "ptr_to_" + makeTypeName(PtrTy->getElementType());
  } else if (auto *IntTy = llvm::dyn_cast<llvm::IntegerType>(Ty)) {
    Name = "i" + std::to_string(IntTy->getBitWidth());
  } else if (auto *StrucTy = llvm::dyn_cast<llvm::StructType>(Ty)) {
    Name = "struct_" + makeCIdentifier(Ty->getStructName().str());
  } else if (auto *FunTy = llvm::dyn_cast<llvm::FunctionType>(Ty)) {
    Name = "func_" + makeTypeName(FunTy->getReturnType());
    if (not FunTy->params().empty()) {
      Name += "_args";
      for (const auto &ArgT : FunTy->params())
        Name += "_" + makeTypeName(ArgT);
    }
  } else if (Ty->isVoidTy()) {
    Name += "void";
  } else {
    revng_unreachable("cannot build Type name");
  }
  return Name;
}

std::string makeSCEVBarrierName(const llvm::Type *Ty) {
  return "revng_scev_barrier_" + makeTypeName(Ty);
}

bool AddSCEVBarrierPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Lifted))
    return false;

  // If the MarkForSerializationPass was not executed, we have nothing to do.
  auto *MarkPass = getAnalysisIfAvailable<MarkForSerializationPass>();
  if (not MarkPass)
    return false;

  const auto &MarkMap = MarkPass->getMap();

  llvm::Module *M = F.getParent();
  llvm::IRBuilder<> Builder(M->getContext());

  bool Changed = false;
  for (auto &I : llvm::instructions(&F)) {

    // Check if I was marked for serialization, and skip it if it wasn't marked.
    auto MarkIt = MarkMap.find(&I);
    if (MarkIt == MarkMap.end())
      continue;

    // If it was marked for serialization, see how.
    const SerializationFlags &Flags = MarkIt->second;
    auto *IType = I.getType();
    if (SerializationFlags::mustBeSerialized(Flags) and IType->isIntOrPtrTy()) {

      // Get the SCEV barrier function associated with IType
      auto BarrierName = makeSCEVBarrierName(IType);
      auto SCEVBarrierF = M->getOrInsertFunction(BarrierName, IType, IType);

      // Insert a call to the SCEV barrier right after I. For now the call to
      // barrier has an undef argument, that will be fixed later.
      Builder.SetInsertPoint(I.getParent(), std::next(I.getIterator()));
      auto *Undef = llvm::UndefValue::get(IType);
      auto *Call = Builder.CreateCall(SCEVBarrierF, Undef);

      // Replace all uses of I with the new call.
      I.replaceAllUsesWith(Call);

      // Now Fix the call to use I as argument.
      Call->setArgOperand(0, &I);
      Changed = true;
    }
  }
  return Changed;
}
