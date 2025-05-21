//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

static Logger<> Log{ "make-local-variables" };

struct MakeLocalVariables : public llvm::FunctionPass {
public:
  static char ID;

  MakeLocalVariables() : FunctionPass(ID) {}

  /// Transform allocas and alloca-like function calls into calls to
  /// `LocalVariable(AddressOf())`
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.setPreservesCFG();
  }
};

using llvm::AllocaInst;
using llvm::dyn_cast;

bool MakeLocalVariables::runOnFunction(llvm::Function &F) {

  llvm::SmallVector<llvm::AllocaInst *, 8> ToReplace;

  // Collect instructions that allocate local variables
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *Alloca = dyn_cast<AllocaInst>(&I))
        ToReplace.push_back(Alloca);

  if (ToReplace.empty())
    return false;

  // Get the model and the function metadata cache
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();

  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::Module &M = *F.getParent();
  llvm::IRBuilder<> Builder(LLVMCtx);
  llvm::Type *PtrSizedInteger = getPointerSizedInteger(LLVMCtx, *Model);

  // Initialize function pools
  auto AddressOfPool = FunctionTags::AddressOf.getPool(M);
  auto LocalVarPool = FunctionTags::LocalVariable.getPool(M);

  // Try to initialize a map for the known model types of llvm::Values
  // that are reachable from F. If this fails, we just bail out because we
  // cannot infer any modelGEP in F, if we have no type information to rely
  // on.
  const model::Function *ModelF = llvmToModelFunction(*Model, F);
  revng_assert(ModelF);
  using ModelTypesMap = std::map<const llvm::Value *,
                                 const model::UpcastableType>;
  ModelTypesMap
    KnownTypes = initModelTypesConsideringUses(F,
                                               ModelF,
                                               *Model,
                                               /* PointersOnly */ false);

  for (auto *Alloca : ToReplace) {
    Builder.SetInsertPoint(Alloca);
    llvm::Type *ResultType = Alloca->getType();

    // Compute the allocated type for the alloca
    model::UpcastableType AllocatedType = model::UpcastableType::empty();
    for (const llvm::Use &U : Alloca->uses()) {
      if (auto *Store = dyn_cast<llvm::StoreInst>(U.getUser())) {
        if (Store->getPointerOperandIndex() == U.getOperandNo()) {
          llvm::Value *Stored = Store->getValueOperand();
          auto It = KnownTypes.find(Stored);
          if (It != KnownTypes.end()) {
            if (AllocatedType.isEmpty()) {
              AllocatedType = It->second.copy();
            } else if (*AllocatedType != *It->second) {
              // Found multiple possible types, relying on a fallback instead.
              AllocatedType = model::UpcastableType::empty();
              break;
            }
          }
        }
      }
    }

    if (AllocatedType.isEmpty())
      AllocatedType = llvmIntToModelType(Alloca->getAllocatedType(), *Model);
    revng_assert(!AllocatedType.isEmpty() && AllocatedType->verify());

    // Inject call to LocalVariable
    llvm::IntegerType *PtrSizedInt = getPointerSizedInteger(LLVMCtx, *Model);
    auto *LocalVarFunctionType = getLocalVarType(PtrSizedInt);
    auto *LocalVarFunction = LocalVarPool.get(PtrSizedInt,
                                              LocalVarFunctionType,
                                              "LocalVariable");
    llvm::Constant *ModelTypeString = toLLVMString(AllocatedType, M);
    auto *LocalVarCall = Builder.CreateCall(LocalVarFunction,
                                            { ModelTypeString });

    // Inject a call to AddressOf
    auto LocalVarType = LocalVarCall->getType();
    auto *AddressOfFunctionType = getAddressOfType(PtrSizedInteger,
                                                   LocalVarType);
    auto *AddressOfFunction = AddressOfPool.get({ PtrSizedInteger,
                                                  LocalVarType },
                                                AddressOfFunctionType,
                                                "AddressOf");
    llvm::Instruction *AddressOfCall = Builder.CreateCall(AddressOfFunction,
                                                          { ModelTypeString,
                                                            LocalVarCall });

    AddressOfCall->copyMetadata(*Alloca);
    llvm::Value *ValueToSubstitute = AddressOfCall;

    // LocalVar and AddressOf work on LLVM integers that represent
    // pointers in the binary. If we are actually using these as
    // pointers in LLVM IR, we need to cast them to the appropriate
    // type.
    if (ResultType->isPointerTy())
      ValueToSubstitute = Builder.CreateIntToPtr(AddressOfCall, ResultType);

    revng_assert(ResultType == ValueToSubstitute->getType());

    Alloca->replaceAllUsesWith(ValueToSubstitute);
    Alloca->eraseFromParent();
  }

  return true;
}

char MakeLocalVariables::ID = 0;

static llvm::RegisterPass<MakeLocalVariables> X("make-local-variables",
                                                "Replace all opcodes that "
                                                "declare local variables with "
                                                "a call to LocalVariable.",
                                                false,
                                                false);
