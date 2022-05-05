//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include <algorithm>
#include <cstdint>
#include <utility>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Support/ModelHelpers.h"

#include "../FuncOrCallInst.h"
#include "DLAMakeModelTypes.h"

using namespace model;
using namespace dla;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

static Logger<> Log("dla-update-model-funcs");
static Logger<> ModelLog("dla-dump-model-with-funcs");

using model::PrimitiveTypeKind::Generic;
using model::PrimitiveTypeKind::PointerOrNumber;

// TODO: implement a better way to merge qualified types
/// Check if a type can be narrowed down to finer types.
static bool canBeNarrowedDown(QualifiedType T) {
  return T.isPrimitive(PointerOrNumber) or T.isPrimitive(Generic);
}

static const llvm::Value *toLLVMValue(const llvm::Argument &A) {
  return static_cast<const llvm::Value *>(&A);
}
static const llvm::Value *toLLVMValue(const llvm::Use &A) {
  return A.get();
}

/// If the DLA recovered a more precise type than the already existing one
/// for any of the arguments of a RawFunctionType, update the model accordingly.
template<typename T>
static bool updateRawFuncArgs(model::Binary &Model,
                              RawFunctionType *ModelPrototype,
                              const T &CallOrFunction,
                              const TypeMapT &DLATypes) {
  bool Updated = false;
  auto &ModelArgs = ModelPrototype->Arguments;
  auto LLVMArgs = getArgs(CallOrFunction);

  // Ensure the LLVM function has the expected number of arguments
  uint64_t EffectiveLLVMArgSize = arg_size(CallOrFunction);

  // In case of presence of stack arguments, there's an extra argument
  if (auto *ModelStackArg = ModelPrototype->StackArgumentsType.UnqualifiedType
                              .get()) {
    revng_assert(ModelPrototype->StackArgumentsType.Qualifiers.empty());
    revng_assert(not LLVMArgs.empty());

    // If the ModelStackArgs is an empty struct we have to fill it up, otherwise
    // it's already known and we don't have to mess it up.
    auto *ModelStackStruct = cast<model::StructType>(ModelStackArg);
    if (ModelStackStruct->Fields.empty()) {
      revng_assert(ModelStackStruct->size());
      auto ModelStackSize = *ModelStackStruct->size();

      // The stack argument on llvm IR is the last argument
      const llvm::Value *LLVMStackArg = toLLVMValue(*std::prev(LLVMArgs.end()));

      revng_log(Log,
                "Updating stack arg " << LLVMStackArg->getNameOrAsOperand());
      LayoutTypePtr Key{ LLVMStackArg, LayoutTypePtr::fieldNumNone };

      if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {
        using namespace model;
        QualifiedType DLAStackType = NewTypeIt->second;
        revng_assert(DLAStackType.size());
        auto DLAStackSize = *DLAStackType.size();

        Type *UnqualifiedType = DLAStackType.UnqualifiedType.get();
        auto *DLAStackStruct = dyn_cast<StructType>(UnqualifiedType);
        if (DLAStackType.Qualifiers.empty() and DLAStackStruct != nullptr) {
          // Import all the fields of the DLA struct that fit in
          for (auto &Field : DLAStackStruct->Fields) {
            revng_assert(Field.Type.size());
            if (Field.Offset + *Field.Type.size() > ModelStackSize)
              break;
            ModelStackStruct->Fields.insert(Field);
          }
          revng_log(Log, "Updated fields");
        } else if (DLAStackSize <= ModelStackSize) {
          // Insert a field of the type recovered by DLA at offset 0
          ModelStackStruct->Fields[0].Type = DLAStackType;
          revng_log(Log,
                    "Updated: inserted fields at offest 0 with ID: "
                      << UnqualifiedType->ID);
        } else {
          revng_log(Log, "Couldn't import the type into the model");
        }

        Updated = true;
      }
    }

    // Decrease the effective argument size, so the next loop leaves the stack
    // arguments alone.
    EffectiveLLVMArgSize -= 1;
  }

  revng_assert(ModelArgs.size() == EffectiveLLVMArgSize);

  revng_log(Log, "Updating args");

  for (const auto &[ModelArg, LLVMArg] : llvm::zip_first(ModelArgs, LLVMArgs)) {
    revng_assert(ModelArg.Type.isScalar());

    const llvm::Value *LLVMVal = toLLVMValue(LLVMArg);
    revng_log(Log, "Updating arg " << LLVMVal->getNameOrAsOperand());

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have
    // no accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this
    // stage.
    if (canBeNarrowedDown(ModelArg.Type)) {
      LayoutTypePtr Key{ LLVMVal, LayoutTypePtr::fieldNumNone };
      if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {

        auto OldSize = *ModelArg.Type.size();
        ModelArg.Type.UnqualifiedType = NewTypeIt->second.UnqualifiedType;
        // The type is associated to a LayoutTypeSystemPtr, hence we have to add
        // the pointer qualifier
        using model::Architecture::getPointerSize;
        size_t PointerBytes = getPointerSize(Model.Architecture);
        auto PointerQual = model::Qualifier::createPointer(PointerBytes);
        ModelArg.Type.Qualifiers.push_back(PointerQual);
        revng_assert(*ModelArg.Type.size() == OldSize);
        Updated = true;
        revng_log(Log,
                  "Updated to " << ModelArg.Type.UnqualifiedType.get()->ID);
      }
    }
  }

  return Updated;
}

static bool updateFuncStackFrame(model::Function &ModelFunc,
                                 const llvm::Function &LLVMFunc,
                                 const TypeMapT &DLATypes,
                                 model::Binary &Model) {
  bool Updated = false;

  if (not ModelFunc.StackFrameType.isValid())
    return Updated;

  auto *OldModelStackFrameType = ModelFunc.StackFrameType.get();
  if (not OldModelStackFrameType)
    return Updated;

  auto *OldStackFrameStruct = cast<model::StructType>(OldModelStackFrameType);
  if (not OldStackFrameStruct->Fields.empty())
    return Updated;

  bool Found = false;
  for (const auto &I : llvm::instructions(LLVMFunc)) {

    auto *Call = dyn_cast<llvm::CallInst>(&I);
    if (not Call)
      continue;

    auto *Callee = Call->getCalledFunction();
    if (not Callee or Callee->getName() != "revng_stack_frame")
      continue;

    revng_assert(not Found, "Multiple calls to revng_stack_frame");
    Found = true;

    revng_log(Log, "Updating stack for " << LLVMFunc.getName());
    LoggerIndent Indent{ Log };
    revng_log(Log, "Was " << OldModelStackFrameType->ID);

    LayoutTypePtr Key{ Call, LayoutTypePtr::fieldNumNone };
    if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {
      model::QualifiedType NewStackType = NewTypeIt->second;
      NewStackType = peelConstAndTypedefs(NewStackType);
      bool IsPointerOrArray = not NewStackType.Qualifiers.empty();

      auto NewStackSize = *NewStackType.size();
      uint64_t OldStackSize = *OldStackFrameStruct->size();

      revng_assert(NewStackSize > 0
                   and NewStackSize < std::numeric_limits<int64_t>::max());

      auto *UnqualNewStack = NewStackType.UnqualifiedType.get();
      if (IsPointerOrArray or isa<model::PrimitiveType>(UnqualNewStack)
          or isa<model::EnumType>(UnqualNewStack)) {
        revng_assert(NewStackSize <= OldStackSize);

        // OldStackFrameStruct is an empty struct, just add the new stack type
        // as the first field
        OldStackFrameStruct->Fields[0].Type = NewStackType;

      } else if (auto *NewS = dyn_cast<model::StructType>(UnqualNewStack)) {
        // If DLA recoverd a new stack size that is too large, we have to shrink
        // it. For now the shrinking does not look deeply inside the types, only
        // at step one into the fields of the struct. We drop all the fields
        // that go over the OldStackSize.
        if (NewStackSize > OldStackSize) {
          const auto IsTooLarge = [OldStackSize](const model::StructField &F) {
            return (F.Offset + *F.Type.size()) > OldStackSize;
          };

          auto It = llvm::find_if(NewS->Fields, IsTooLarge);
          auto End = NewS->Fields.end();
          NewS->Fields.erase(It, End);
        }

        for (const auto &Field : NewS->Fields)
          OldStackFrameStruct->Fields.insert(Field);

      } else if (auto *NewU = dyn_cast<model::UnionType>(UnqualNewStack)) {
        // Prepare the new stack type, which cannot be a union, it has to be
        // a model::StructType
        model::TypePath NewStackStruct = createEmptyStruct(Model, OldStackSize);

        // If DLA recoverd a new stack size that is too large, we have to shrink
        // it. For now the shrinking does not look deeply inside the types, only
        // at step one into the fields of the union. We drop all the fields that
        // are larger than OldStackSize.
        auto FieldsRemaining = NewU->Fields.size();
        if (NewStackSize > OldStackSize) {

          const auto IsTooLarge = [OldStackSize](const model::UnionField &F) {
            return *F.Type.size() > OldStackSize;
          };

          // First, detect the fields that are too large.
          llvm::SmallSet<size_t, 8> FieldsToDrop;
          auto It = NewU->Fields.begin();
          auto End = NewU->Fields.end();
          for (; It != End; ++It) {
            revng_assert(It->verify());
            revng_assert(It->Type.verify());
            if (IsTooLarge(*It))
              FieldsToDrop.insert(It->Index);
          }

          revng_assert(not FieldsToDrop.empty());

          // Count the fields that are left in the union if we remove those
          // selected to be dropped.
          FieldsRemaining = NewU->Fields.size() - FieldsToDrop.size();

          // If we need to drop at least one field but not all of them, we have
          // to renumber the remaining fields, otherwise the union will not
          // verify.
          if (FieldsRemaining) {

            // Drop the large fields
            for (auto &FieldNum : FieldsToDrop)
              NewU->Fields.erase(FieldNum);

            // Re-enumerate the remaining fields
            for (auto &Group : llvm::enumerate(NewU->Fields))
              Group.value().Index = Group.index();

            revng_assert(NewU->Fields.size() == FieldsRemaining);
            revng_assert(NewU->verify());
          }
        }

        // If there are fields left in the union, then inject them in the stack
        // struct as fields.
        if (FieldsRemaining) {
          cast<model::StructType>(NewStackStruct.get())
            ->Fields[0]
            .Type = model::QualifiedType(Model.getTypePath(NewU), {});
        }

        ModelFunc.StackFrameType = NewStackStruct;

      } else {
        revng_abort();
      }

      revng_log(Log, "Updated to " << ModelFunc.StackFrameType.get()->ID);

      revng_assert(isa<model::StructType>(ModelFunc.StackFrameType.get()));
      revng_assert(*ModelFunc.StackFrameType.get()->size() == OldStackSize);
      revng_assert(ModelFunc.StackFrameType.get()->verify());

      Updated = true;
    }
  }
  return Updated;
}

/// If the DLA recovered a more precise type than the already existing one
/// for the return type of a RawFunctionType, update the model accordingly.
static bool updateRawFuncRetValue(model::Binary &Model,
                                  RawFunctionType *ModelPrototype,
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

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have
    // no accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this
    // stage.
    auto &ModelRetVal = ModelRet.value();
    if (canBeNarrowedDown(ModelRetVal.Type)) {
      LayoutTypePtr Key{ LLVMRetVal, Index };
      if (auto NewTypeIt = TypeMap.find(Key); NewTypeIt != TypeMap.end()) {
        ModelRetVal.Type.UnqualifiedType = NewTypeIt->second.UnqualifiedType;
        // The type is associated to a LayoutTypeSystemPtr, hence we have to
        // add the pointer qualifier
        using model::Architecture::getPointerSize;
        size_t PointerBytes = getPointerSize(Model.Architecture);
        auto PtrQualifier = model::Qualifier::createPointer(PointerBytes);
        ModelRetVal.Type.Qualifiers.push_back(PtrQualifier);
        Updated = true;
        revng_log(Log,
                  "Updated to " << ModelRetVal.Type.UnqualifiedType.get()->ID);
      }
    }
  }

  return Updated;
}

/// Update the prototype of a function with the types recovered by DLA.
template<typename T>
static bool updateFuncPrototype(model::Binary &Model,
                                model::Type *Prototype,
                                const T &CallOrFunction,
                                const TypeMapT &TypeMap) {
  revng_assert(Prototype);
  bool Updated = false;

  revng_log(Log, "Updating func prototype");

  if (auto *RawPrototype = dyn_cast<RawFunctionType>(Prototype)) {
    Updated |= updateRawFuncArgs(Model, RawPrototype, CallOrFunction, TypeMap);
    Updated |= updateRawFuncRetValue(Model,
                                     RawPrototype,
                                     CallOrFunction,
                                     getRetType(CallOrFunction),
                                     TypeMap);
  } else {
    revng_abort("CABIFunctionTypes not yet supported");
  }

  return Updated;
}

bool dla::updateFuncSignatures(const llvm::Module &M,
                               TupleTree<model::Binary> &Model,
                               const TypeMapT &TypeMap) {
  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-before-func-update.yaml");
  if (VerifyLog.isEnabled())
    revng_assert(Model->verify());

  bool Updated = false;

  for (const auto &LLVMFunc : M.functions()) {
    // Update the function's prototype
    model::Function *ModelFunc = llvmToModelFunction(*Model.get(), LLVMFunc);
    if (not ModelFunc)
      continue;

    Type *ModelPrototype = ModelFunc->Prototype.get();
    revng_log(Log,
              "Updating prototype of function "
                << LLVMFunc.getNameOrAsOperand());
    Updated |= updateFuncPrototype(*Model, ModelPrototype, &LLVMFunc, TypeMap);
    Updated |= updateFuncStackFrame(*ModelFunc, LLVMFunc, TypeMap, *Model);

    // Update prototypes associated to indirect calls, if any are found
    for (const auto &Inst : LLVMFunc)
      if (const auto *I = llvm::dyn_cast<llvm::CallInst>(&Inst)) {
        auto Prototype = getCallSitePrototype(*Model.get(), I, ModelFunc);
        if (Prototype.isValid()) {
          revng_log(Log,
                    "Updating prototype of indirect call "
                      << I->getNameOrAsOperand());
          Updated |= updateFuncPrototype(*Model, Prototype.get(), I, TypeMap);
        }
      }
  }

  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-after-func-update.yaml");
  if (VerifyLog.isEnabled())
    revng_assert(Model->verify());

  return Updated;
}
