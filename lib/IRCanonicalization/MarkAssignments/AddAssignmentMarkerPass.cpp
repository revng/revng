//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

/// Pass that detects Instructions in a Functions for which we have to generate
/// a variable assignment when decompiling to C, and wraps them in special
/// marker calls.

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/Mangling.h"
#include "revng-c/Support/ModelHelpers.h"

#include "MarkAssignments.h"

struct AddAssignmentMarkersPass : public llvm::FunctionPass {
public:
  static char ID;

  AddAssignmentMarkersPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<FunctionMetadataCachePass>();
  }

  bool runOnFunction(llvm::Function &F) override;
};

bool AddAssignmentMarkersPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  MarkAssignments::AssignmentMap
    Assignments = MarkAssignments::selectAssignments(F);

  if (Assignments.empty())
    return false;

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const TupleTree<model::Binary> &Model = ModelWrapper.getReadOnlyModel();

  auto ModelFunction = llvmToModelFunction(*Model, F);
  revng_assert(ModelFunction != nullptr);
  auto &Cache = getAnalysis<FunctionMetadataCachePass>().get();

  auto TypeMap = initModelTypes(Cache,
                                F,
                                ModelFunction,
                                *Model,
                                /*PointerOnly*/ false);

  llvm::Module *M = F.getParent();
  llvm::IRBuilder<> Builder(M->getContext());
  bool Changed = false;

  OpaqueFunctionsPool<llvm::Type *> LocalVarPool(M, false);
  initLocalVarPool(LocalVarPool);
  OpaqueFunctionsPool<llvm::Type *> AssignPool(M, false);
  initAssignPool(AssignPool);
  OpaqueFunctionsPool<llvm::Type *> CopyPool(M, false);
  initCopyPool(CopyPool);

  std::map<llvm::CallInst *, llvm::CallInst *> StructCallToLocalVarType;

  for (auto &[I, Flag] : Assignments) {
    auto *IType = I->getType();

    // We cannot wrap void-typed things into wrappers.
    // We'll have to handle them in another way in the decompilation pipeline
    if (IType->isVoidTy())
      continue;

    if (IType->isAggregateType()) {
      // This is a call to a function that return a struct on llvm
      // type system. We cannot handle it like the others, because its return
      // type is not on the model (only individual fields are), so we cannot
      // serialize its QualifiedType in the LocalVariable.
      //
      // We'll have to deal with it later in the decompiler backend.
      revng_assert(not TypeMap.contains(I));
      continue;
    }

    if (bool(Flag)) {

      // We should never be creating a LocalVariable for a reference, since
      // we cannot express them in C.
      revng_assert(not isCallToTagged(I, FunctionTags::IsRef),
                   dumpToString(I).c_str());

      // First, we have to declare the LocalVariable, in the correct place, i.e.
      // either the entry block or just before I.
      if (needsTopScopeDeclaration(*I))
        Builder.SetInsertPoint(&F.getEntryBlock().front());
      else
        Builder.SetInsertPoint(I);

      auto *LocalVarFunctionType = getLocalVarType(IType);
      auto *LocalVarFunction = LocalVarPool.get(IType,
                                                LocalVarFunctionType,
                                                "LocalVariable");

      // Compute the model type returned from the call.
      llvm::Constant *ModelTypeString = serializeToLLVMString(TypeMap.at(I),
                                                              *M);

      // Inject call to LocalVariable
      auto *LocalVarCall = Builder.CreateCall(LocalVarFunction,
                                              { ModelTypeString });

      // Then, we have to replace all the uses of I so that they make a Copy
      // from the new LocalVariable
      for (llvm::Use &U : llvm::make_early_inc_range(I->uses())) {
        revng_assert(isa<llvm::Instruction>(U.getUser()));
        Builder.SetInsertPoint(cast<llvm::Instruction>(U.getUser()));

        // Create a Copy to dereference the LocalVariable
        auto *CopyFnType = getCopyType(LocalVarCall->getType());
        auto *CopyFunction = CopyPool.get(LocalVarCall->getType(),
                                          CopyFnType,
                                          "Copy");
        auto *CopyCall = Builder.CreateCall(CopyFunction, { LocalVarCall });
        U.set(CopyCall);
      }

      // Finally, we have to assign the result of I to the local variable, right
      // after I itself.
      Builder.SetInsertPoint(I->getParent(), std::next(I->getIterator()));

      // Inject Assign() function
      auto *AssignFnType = getAssignFunctionType(IType,
                                                 LocalVarCall->getType());
      auto *AssignFunction = AssignPool.get(IType, AssignFnType, "Assign");

      Builder.CreateCall(AssignFunction, { I, LocalVarCall });

      Changed = true;
    }
  }

  return Changed;
}

char AddAssignmentMarkersPass::ID = 0;

using Register = llvm::RegisterPass<AddAssignmentMarkersPass>;
static Register X("add-assignment-markers",
                  "Pass that adds assignment markers to the IR",
                  false,
                  false);
