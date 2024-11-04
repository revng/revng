//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

struct RemoveLoadStore : public llvm::FunctionPass {
public:
  static char ID;

  RemoveLoadStore() : FunctionPass(ID) {}

  /// Replace all `load` instructions with calls to `Copy(ModelGEP())` and all
  /// `store` instructions with calls to `Assign(ModelGEP())`.
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.setPreservesCFG();
  }
};

using llvm::dyn_cast;

static llvm::CallInst *buildDerefCall(llvm::Module &M,
                                      llvm::IRBuilder<> &Builder,
                                      llvm::Value *Arg,
                                      const model::UpcastableType &PointedType,
                                      llvm::Type *ReturnType) {
  llvm::Type *BaseType = Arg->getType();

  auto *ModelGEPFunction = getModelGEP(M, ReturnType, BaseType);

  // The first argument is always a pointer to a constant global variable
  // that holds the string representing the yaml serialization of the base type
  // of the modelGEP
  auto *BaseTypeConstantStrPtr = toLLVMString(PointedType, M);

  // The second argument is the base address, and the third (representing the
  // array access) is defaulted to 0, representing regular pointer access (not
  // array access).
  auto *Int64Type = llvm::IntegerType::getIntNTy(M.getContext(), 64);
  auto *Zero = llvm::ConstantInt::get(Int64Type, 0);
  llvm::CallInst *InjectedCall = Builder.CreateCall(ModelGEPFunction,
                                                    { BaseTypeConstantStrPtr,
                                                      Arg,
                                                      Zero });

  return InjectedCall;
}

bool RemoveLoadStore::runOnFunction(llvm::Function &F) {

  // Get the model
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();

  // Collect model types
  auto TypeMap = initModelTypes(F,
                                llvmToModelFunction(*Model, F),
                                *Model,
                                /* PointersOnly = */ false);

  // Initialize the IR builder to inject functions
  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::Module &M = *F.getParent();
  llvm::IRBuilder<> Builder(LLVMCtx);

  // Initialize function pool
  auto AssignPool = FunctionTags::Assign.getPool(M);
  auto CopyPool = FunctionTags::Copy.getPool(M);

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
        const model::UpcastableType &PointedT = TypeMap.at(Load);

        // Check that the Model type is compatible with the Load size
        revng_assert(areMemOpCompatible(*PointedT, *Load->getType(), *Model));

        // Create an index-less ModelGEP for the pointer operand
        auto *DerefCall = buildDerefCall(M,
                                         Builder,
                                         PtrOp,
                                         PointedT,
                                         Load->getType());

        // Create a Copy to dereference the ModelGEP
        auto *CopyFnType = getCopyType(DerefCall->getType());
        auto *CopyFunction = CopyPool.get(DerefCall->getType(),
                                          CopyFnType,
                                          "Copy");
        InjectedCall = Builder.CreateCall(CopyFunction, { DerefCall });

        // Add the dereferenced type to the type map
        auto [_, Inserted] = TypeMap.insert({ InjectedCall, PointedT.copy() });
        revng_assert(Inserted);

      } else if (auto *Store = dyn_cast<llvm::StoreInst>(&I)) {
        llvm::Value *ValueOp = Store->getValueOperand();
        llvm::Value *PointerOp = Store->getPointerOperand();
        llvm::Type *PointedType = ValueOp->getType();

        const model::Type &PointerOpType = *TypeMap.at(PointerOp);
        model::UpcastableType StoredType = TypeMap.at(ValueOp);

        // Use the model information coming from pointer operand only if the
        // size is the same as the store's original size.
        if (PointerOpType.isPointer()) {
          const model::Type &PointeeType = PointerOpType.getPointee();

          // We want to make sure that the StoredType is compatible with the
          // place it is being stored into.
          // However, we have to work around string literals, because they are
          // custom opcodes that return an integer on LLVM IR, but that
          // integers always represents a pointer in the decompiled C code.
          // TODO: this can actually be improved upon.
          // Probably MakeSegmentRefPass (who injects calls to cstringLiteral)
          // should emit an AddressOf, and cstringLiteral should have reference
          // semantics. If we do this it has to be integrated with DLA.
          if (isCallToTagged(ValueOp, FunctionTags::StringLiteral))
            PointedType = llvm::PointerType::getUnqual(Store->getContext());

          if (areMemOpCompatible(PointeeType, *PointedType, *Model))
            StoredType = PointeeType;
        }
        revng_assert(areMemOpCompatible(*StoredType, *PointedType, *Model));

        auto *DerefCall = buildDerefCall(M,
                                         Builder,
                                         PointerOp,
                                         StoredType,
                                         ValueOp->getType());

        // Add the dereferenced type to the type map
        TypeMap.insert({ DerefCall, StoredType });

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
