//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <algorithm>
#include <cstdint>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAHelpers.h"
#include "../DLAModelFuncHelpers.h"
#include "DLAMakeModelTypes.h"

using namespace model;
using namespace dla;

static Logger<> Log("dla-update-model-funcs");
static Logger<> ModelLog("dla-dump-model-with-funcs");

using WritableModelT = TupleTree<model::Binary>;

using model::PrimitiveTypeKind::Generic;
using model::PrimitiveTypeKind::PointerOrNumber;
using model::QualifierKind::Pointer;

// TODO: implement a better way to merge qualified types
///\brief Check if a type can be narrowed down to finer types.
static bool canBeNarrowedDown(QualifiedType T) {
  return T.isPrimitive(PointerOrNumber) or T.isPrimitive(Generic);
}

static const llvm::Value *toLLVMValue(const llvm::Argument &A) {
  return static_cast<const llvm::Value *>(&A);
}
static const llvm::Value *toLLVMValue(const llvm::Use &A) {
  return A.get();
}

///\brief If the DLA recovered a more precise type than the already existing one
/// for any of the arguments of a RawFunctionType, update the model accordingly.
template<typename T>
static bool updateRawFuncArgs(RawFunctionType *ModelPrototype,
                              const T &CallOrFunction,
                              const TypeMapT &DLATypes) {
  bool Updated = false;
  auto &ModelArgs = ModelPrototype->Arguments;
  auto LLVMArgs = getArgs(CallOrFunction);
  uint64_t LLVMArgSize = llvm::size(LLVMArgs);
  revng_assert(ModelArgs.size() == LLVMArgSize);

  revng_log(Log, "Updating args");

  for (const auto &[ModelArg, LLVMArg] : llvm::zip(ModelArgs, LLVMArgs)) {
    revng_assert(ModelArg.Type.isScalar());

    const llvm::Value *LLVMVal = toLLVMValue(LLVMArg);
    revng_log(Log, "Updating arg " << LLVMVal->getNameOrAsOperand());
    LayoutTypePtr Key{ LLVMVal, LayoutTypePtr::fieldNumNone };
    auto NewTypeIt = DLATypes.find(Key);

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have no
    // accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this stage.
    if (canBeNarrowedDown(ModelArg.Type) and NewTypeIt != DLATypes.end()) {
      auto PtrSize = ModelArg.Type.size();
      ModelArg.Type.UnqualifiedType = NewTypeIt->second.UnqualifiedType;
      // The type is associated to a LayoutTypeSystemPtr, hence we have to add
      // the pointer qualifier
      ModelArg.Type.Qualifiers.push_back({ Pointer, PtrSize.value_or(0) });
      Updated = true;
      revng_log(Log, "Updated to " << ModelArg.Type.UnqualifiedType.get()->ID);
    }
  }

  return Updated;
}

///\brief If the DLA recovered a more precise type than the already existing one
/// for the return type of a RawFunctionType, update the model accordingly.
static bool updateRawFuncRetValue(RawFunctionType *ModelPrototype,
                                  const llvm::Value *LLVMRetVal,
                                  const llvm::Type *LLVMRetType,
                                  const TypeMapT &TypeMap) {
  bool Updated = false;
  auto &ModelRetVals = ModelPrototype->ReturnValues;
  revng_log(Log, "Updating return values");

  if (LLVMRetType->isIntOrPtrTy()) {
    revng_assert(ModelRetVals.size() == 1);
  } else if (LLVMRetType->isVoidTy()) {
    revng_assert(ModelRetVals.size() == 0);
  } else {
    const auto *RetValStruct = llvm::cast<llvm::StructType>(LLVMRetType);
    const auto &SubTypes = RetValStruct->subtypes();
    revng_assert(SubTypes.size() == ModelRetVals.size());

    auto IsScalar = [](const llvm::Type *E) { return E->isSingleValueType(); };
    revng_assert(llvm::all_of(SubTypes, IsScalar));
  }

  const bool IsScalar = ModelRetVals.size() == 1;
  for (auto &ModelRet : llvm::enumerate(ModelRetVals)) {

    unsigned int Index = IsScalar ? LayoutTypePtr::fieldNumNone :
                                    ModelRet.index();
    revng_log(Log,
              "Updating elem " << Index << " of "
                               << LLVMRetVal->getNameOrAsOperand());
    LayoutTypePtr Key{ LLVMRetVal, Index };
    auto NewTypeIt = TypeMap.find(Key);

    auto &ModelRetVal = ModelRet.value();
    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have no
    // accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this stage.
    if (canBeNarrowedDown(ModelRetVal.Type) and NewTypeIt != TypeMap.end()) {
      auto PtrSize = ModelRetVal.Type.size();
      ModelRetVal.Type.UnqualifiedType = NewTypeIt->second.UnqualifiedType;
      // The type is associated to a LayoutTypeSystemPtr, hence we have to add
      // the pointer qualifier
      ModelRetVal.Type.Qualifiers.push_back({ Pointer, PtrSize.value_or(0) });
      Updated = true;
      revng_log(Log,
                "Updated to " << ModelRetVal.Type.UnqualifiedType.get()->ID);
    }
  }

  return Updated;
}

///\brief Update the prototype of a function with the types recovered by DLA.
template<typename T>
static bool updateFuncPrototype(model::Type *Prototype,
                                const T &CallOrFunction,
                                const TypeMapT &TypeMap) {
  revng_assert(Prototype);
  bool Updated = false;

  revng_log(Log, "Updating func prototype");

  if (auto *RawPrototype = llvm::dyn_cast<RawFunctionType>(Prototype)) {
    Updated |= updateRawFuncArgs(RawPrototype, CallOrFunction, TypeMap);
    Updated |= updateRawFuncRetValue(RawPrototype,
                                     CallOrFunction,
                                     getRetType(CallOrFunction),
                                     TypeMap);
  } else {
    revng_abort("CABIFunctionTypes not yet supported");
  }

  return Updated;
}

bool dla::updateFuncSignatures(const llvm::Module &M,
                               WritableModelT &Model,
                               const TypeMapT &TypeMap) {
  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-before-func-update.yaml");
  if (VerifyLog.isEnabled())
    revng_assert(Model->verify(true));

  bool Updated = false;

  for (const auto &LLVMFunc : M.functions()) {
    const auto &FuncMetaAddr = getMetaAddress(&LLVMFunc);
    if (not FuncMetaAddr.isValid())
      continue;

    // Update the function's prototype
    auto &ModelFunc = Model->Functions.at(FuncMetaAddr);
    Type *ModelPrototype = ModelFunc.Prototype.get();
    revng_log(Log,
              "Updating prototype of function "
                << LLVMFunc.getNameOrAsOperand());
    Updated |= updateFuncPrototype(ModelPrototype, &LLVMFunc, TypeMap);

    // Update prototypes associated to indirect calls, if any are found
    for (const auto &Inst : LLVMFunc)
      if (const auto *I = llvm::dyn_cast<llvm::CallInst>(&Inst))
        if (auto *Prototype = getIndirectCallPrototype(I, *Model.get())) {
          revng_log(Log,
                    "Updating prototype of indirect call "
                      << I->getNameOrAsOperand());
          Updated |= updateFuncPrototype(Prototype, I, TypeMap);
        }
  }

  if (VerifyLog.isEnabled())
    revng_assert(Model->verify(true));
  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-after-func-update.yaml");

  return Updated;
}
