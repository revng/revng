//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

struct RemoveLoadStore : public llvm::FunctionPass {
public:
  static char ID;

  RemoveLoadStore() : FunctionPass(ID) {}

  /// Replace all `load` instructions with calls to `ModelGEP()` and all
  /// `store` instructions with calls to `Assign(ModelGEP())`.
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.addRequired<FunctionMetadataCachePass>();
    AU.setPreservesCFG();
  }
};

using llvm::CallInst;
using llvm::dyn_cast;
using llvm::Type;
using model::QualifiedType;

static llvm::CallInst *buildDerefCall(llvm::Module &M,
                                      llvm::IRBuilder<> &Builder,
                                      llvm::Value *Arg,
                                      model::QualifiedType &PointedType,
                                      llvm::Type *ReturnType) {
  llvm::Type *BaseType = Arg->getType();

  auto *ModelGEPFunction = getModelGEP(M, ReturnType, BaseType);

  // The first argument is always a pointer to a constant global variable
  // that holds the string representing the yaml serialization of the
  // qualified type of the base type of the modelGEP
  auto *BaseTypeConstantStrPtr = serializeToLLVMString(PointedType, M);

  // The second argument is the base address
  llvm::CallInst *InjectedCall = Builder.CreateCall(ModelGEPFunction,
                                                    { BaseTypeConstantStrPtr,
                                                      Arg });

  return InjectedCall;
}

bool RemoveLoadStore::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Get the model
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();

  // Collect model types
  auto TypeMap = initModelTypes(getAnalysis<FunctionMetadataCachePass>().get(),
                                F,
                                llvmToModelFunction(*Model, F),
                                *Model,
                                /*PointersOnly=*/false);

  // Initialize the IR builder to inject functions
  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::Module &M = *F.getParent();
  llvm::IRBuilder<> Builder(LLVMCtx);

  // Initialize function pool
  OpaqueFunctionsPool<llvm::Type *> AssignPool(&M, false);
  initAssignPool(AssignPool);
  OpaqueFunctionsPool<llvm::Type *> CopyPool(&M, false);
  initCopyPool(CopyPool);

  llvm::SmallVector<llvm::Instruction *, 32> ToRemove;

  // Make replacements
  for (auto &BB : F) {
    auto CurInst = BB.begin();
    while (CurInst != BB.end()) {
      auto NextInst = std::next(CurInst);
      auto &I = *CurInst;

      // Consider only load and store instruction
      if (not isa<llvm::LoadInst>(&I) and not isa<llvm::StoreInst>(&I)) {
        CurInst = NextInst;
        continue;
      }

      Builder.SetInsertPoint(&I);
      ToRemove.push_back(&I);

      llvm::CallInst *InjectedCall = nullptr;

      if (auto *Load = dyn_cast<llvm::LoadInst>(&I)) {
        llvm::Value *PtrOp = Load->getPointerOperand();
        QualifiedType PointedType = TypeMap.at(Load);

        // Check that the Model type is compatible with the Load size
        revng_assert(areMemOpCompatible(PointedType, *Load->getType(), *Model));

        // Create an index-less ModelGEP for the pointer operand
        auto *DerefCall = buildDerefCall(M,
                                         Builder,
                                         PtrOp,
                                         PointedType,
                                         Load->getType());

        // Create a Copy to dereference the ModelGEP
        auto *CopyFnType = getCopyType(DerefCall->getType());
        auto *CopyFunction = CopyPool.get(DerefCall->getType(),
                                          CopyFnType,
                                          "Copy");
        InjectedCall = Builder.CreateCall(CopyFunction, { DerefCall });

        // Add the dereferenced type to the type map
        auto [_, Inserted] = TypeMap.insert({ InjectedCall, PointedType });
        revng_assert(Inserted);

      } else if (auto *Store = dyn_cast<llvm::StoreInst>(&I)) {
        llvm::Value *ValueOp = Store->getValueOperand();
        llvm::Value *PointerOp = Store->getPointerOperand();
        llvm::Type *PointedType = ValueOp->getType();

        QualifiedType PointerOpQT = TypeMap.at(PointerOp);
        QualifiedType StoredQT;

        // Use the model information coming from pointer operand only if the
        // size is the same as the store's original size.
        if (PointerOpQT.isPointer()) {
          model::QualifiedType PointedQT = dropPointer(PointerOpQT);

          if (areMemOpCompatible(PointedQT, *ValueOp->getType(), *Model))
            StoredQT = PointedQT;
        }

        // If the pointer operand does not have a pointer type in the model,
        // it means we could not recover type information about the pointer
        // operand. Hence, we need to use the stored operand to understand the
        // type pointed by the `Store` instruction.
        if (not StoredQT.UnqualifiedType().isValid())
          StoredQT = TypeMap.at(ValueOp);

        revng_assert(areMemOpCompatible(StoredQT, *ValueOp->getType(), *Model));

        auto *DerefCall = buildDerefCall(M,
                                         Builder,
                                         PointerOp,
                                         StoredQT,
                                         PointedType);

        // Add the dereferenced type to the type map
        TypeMap.insert({ DerefCall, StoredQT });

        // Inject Assign() function
        auto *AssignFnType = getAssignFunctionType(ValueOp->getType(),
                                                   DerefCall->getType());
        auto *AssignFunction = AssignPool.get(ValueOp->getType(),
                                              AssignFnType,
                                              "Assign");
        InjectedCall = Builder.CreateCall(AssignFunction,
                                          { ValueOp, DerefCall });
      }

      // Replace original Instruction
      revng_assert(InjectedCall);
      I.replaceAllUsesWith(InjectedCall);

      CurInst = NextInst;
    }
  }

  if (ToRemove.empty())
    return false;

  // Remove all load/store instructions that have been substituted
  for (auto *InstToRemove : ToRemove)
    InstToRemove->eraseFromParent();

  return true;
}

char RemoveLoadStore::ID = 0;

static llvm::RegisterPass<RemoveLoadStore> X("remove-load-store",
                                             "Replaces all loads and stores "
                                             "with ModelGEP() and assign() "
                                             "calls.",
                                             false,
                                             false);
