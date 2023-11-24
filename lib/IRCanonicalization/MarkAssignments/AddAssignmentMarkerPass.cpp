//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

/// Pass that detects Instructions in a Functions for which we have to generate
/// a variable assignment when decompiling to C, and wraps them in special
/// marker calls.

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/Architecture.h"
#include "revng/Model/Generated/Early/Architecture.h"
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

using namespace llvm;

struct AddAssignmentMarkersPass : public FunctionPass {
public:
  static char ID;

  AddAssignmentMarkersPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<FunctionMetadataCachePass>();
  }

  bool runOnFunction(Function &F) override;
};

bool AddAssignmentMarkersPass::runOnFunction(Function &F) {

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

  Module *M = F.getParent();
  IRBuilder<> Builder(M->getContext());
  bool Changed = false;

  OpaqueFunctionsPool<Type *> LocalVarPool(M, false);
  initLocalVarPool(LocalVarPool);
  OpaqueFunctionsPool<Type *> AssignPool(M, false);
  initAssignPool(AssignPool);
  OpaqueFunctionsPool<Type *> CopyPool(M, false);
  initCopyPool(CopyPool);

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

      // If we're trying to create a new local variable for a copy of another
      // existing local variable, just reuse the existing one.
      if (auto *CallToCopy = getCallToTagged(I, FunctionTags::Copy))
        if (auto *CallToLocal = getCallToTagged(CallToCopy->getArgOperand(0),
                                                FunctionTags::LocalVariable))
          continue;

      // First, we have to declare the LocalVariable, always at the entry block.
      Builder.SetInsertPoint(&F.getEntryBlock().front());

      auto *LocalVarFunctionType = getLocalVarType(IType);
      auto *LocalVarFunction = LocalVarPool.get(IType,
                                                LocalVarFunctionType,
                                                "LocalVariable");

      // Compute the model type returned from the call.
      model::QualifiedType VariableType = TypeMap.at(I);
      const llvm::DataLayout &DL = I->getModule()->getDataLayout();
      auto ModelSize = VariableType.size().value();
      auto IRSize = DL.getTypeAllocSize(IType);
      if (ModelSize >= IRSize) {
        if (ModelSize > IRSize) {
          auto Prototype = Cache.getCallSitePrototype(*Model,
                                                      cast<CallInst>(I));
          using namespace abi::FunctionType;
          abi::FunctionType::Layout Layout = Layout::make(*Prototype.get());
          revng_assert(Layout.hasSPTAR());
          VariableType = llvmIntToModelType(IType, *Model);
        }
      } else {
        revng_assert(IType->isPointerTy());
        using model::Architecture::getPointerSize;
        auto PtrSize = getPointerSize(Model->Architecture());
        revng_assert(ModelSize == PtrSize);
      }

      // TODO: until we don't properly handle variable declarations with inline
      // initialization (might need MLIR), we cannot declare const local
      // variables, because their initialization (which is forcibly out-of-line)
      // would assign them and fail to compile.
      // For this reason if at this point we're trying to declare a constant
      // local variable, we're forced to throw away the const-qualifier.
      if (VariableType.isConst())
        VariableType = getNonConst(VariableType);

      Constant *ModelTypeString = serializeToLLVMString(VariableType, *M);

      // Inject call to LocalVariable
      CallInst *LocalVarCall = Builder.CreateCall(LocalVarFunction,
                                                  { ModelTypeString });

      // Then, we have to replace all the uses of I so that they make a Copy
      // from the LocalVariable
      for (Use &U : llvm::make_early_inc_range(I->uses())) {
        revng_assert(isa<Instruction>(U.getUser()));
        Builder.SetInsertPoint(cast<Instruction>(U.getUser()));

        // Create a Copy to dereference the LocalVariable
        auto *CopyFnType = getCopyType(LocalVarCall->getType());
        auto *CopyFunction = CopyPool.get(LocalVarCall->getType(),
                                          CopyFnType,
                                          "Copy");
        auto *CopyCall = Builder.CreateCall(CopyFunction, { LocalVarCall });
        U.set(CopyCall);
      }

      // We have to assign the result of I to the local variable, right
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

using Register = RegisterPass<AddAssignmentMarkersPass>;
static Register X("add-assignment-markers",
                  "Pass that adds assignment markers to the IR",
                  false,
                  false);
