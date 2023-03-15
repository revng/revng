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
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
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

/// Returns true if \a T can be upgraded by DLA
// TODO: implement a better way to merge qualified types, based on the lattice
// of primitive types.
static bool canBeUpgraded(QualifiedType T) {
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
// clang-format off
requires std::same_as<std::remove_const_t<T>, llvm::CallInst> or
         std::same_as<std::remove_const_t<T>, llvm::Function>
// clang-format on
static bool updateArgumentTypes(model::Binary &Model,
                                RawFunctionType *ModelPrototype,
                                const T *CallOrFunction,
                                const TypeMapT &DLATypes) {
  bool Updated = false;
  auto &ModelArgs = ModelPrototype->Arguments();
  auto LLVMArgs = getArgs(CallOrFunction);

  uint64_t EffectiveLLVMArgSize = arg_size(CallOrFunction);

  // In case of presence of stack arguments, there's an extra argument
  if (not ModelPrototype->StackArgumentsType().UnqualifiedType().empty()) {
    revng_log(Log, "Updating stack argument");
    auto *ModelStackArg = ModelPrototype->StackArgumentsType()
                            .UnqualifiedType()
                            .get();
    revng_assert(ModelPrototype->StackArgumentsType().Qualifiers().empty());
    revng_assert(not LLVMArgs.empty());

    // If the ModelStackArgs is an empty struct we have to fill it up, otherwise
    // it's already known and we don't have to mess it up.
    auto *ModelStackStruct = cast<model::StructType>(ModelStackArg);
    if (ModelStackStruct->Fields().empty()) {
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

        Type *UnqualifiedType = DLAStackType.UnqualifiedType().get();
        auto *DLAStackStruct = dyn_cast<StructType>(UnqualifiedType);
        if (DLAStackType.Qualifiers().empty() and DLAStackStruct != nullptr) {
          // Import all the fields of the DLA struct that fit in
          for (auto &Field : DLAStackStruct->Fields()) {
            revng_assert(Field.Type().size());
            if (Field.Offset() + *Field.Type().size() > ModelStackSize)
              break;
            ModelStackStruct->Fields().insert(Field);
          }
          revng_log(Log, "Updated fields");
        } else if (DLAStackSize <= ModelStackSize) {
          // Insert a field of the type recovered by DLA at offset 0
          ModelStackStruct->Fields()[0].Type() = DLAStackType;
          revng_log(Log,
                    "Updated: inserted fields at offest 0 with ID: "
                      << UnqualifiedType->ID());
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

  revng_assert(ModelArgs.size() == EffectiveLLVMArgSize
               or (ModelArgs.size() == 1 and ModelArgs.begin()->Type().isVoid()
                   and EffectiveLLVMArgSize == 0));

  revng_log(Log, "Updating register arguments");

  for (const auto &[ModelArg, LLVMArg] : llvm::zip_first(ModelArgs, LLVMArgs)) {
    revng_assert(ModelArg.Type().isScalar());

    const llvm::Value *LLVMVal = toLLVMValue(LLVMArg);
    revng_log(Log, "Updating argument " << LLVMVal->getNameOrAsOperand());

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have
    // no accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this
    // stage.
    if (canBeUpgraded(ModelArg.Type())) {
      LayoutTypePtr Key{ LLVMVal, LayoutTypePtr::fieldNumNone };
      if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {
        auto OldSize = *ModelArg.Type().size();
        // The type is associated to a LayoutTypeSystemPtr, hence we have to add
        // the pointer qualifier
        ModelArg.Type() = NewTypeIt->second.getPointerTo(Model.Architecture());
        revng_assert(*ModelArg.Type().size() == OldSize);
        revng_log(Log,
                  "Updated to "
                    << ModelArg.Type().UnqualifiedType().get()->ID());
        Updated = true;
      }
    }
  }

  return Updated;
}

/// If the DLA recovered a more precise type than the already existing one
/// for the return type of a RawFunctionType, update the model accordingly.
static bool updateReturnType(model::Binary &Model,
                             RawFunctionType *ModelPrototype,
                             const llvm::Value *LLVMRetVal,
                             const llvm::Type *LLVMRetType,
                             const TypeMapT &TypeMap) {
  bool Updated = false;
  auto &ModelRetVals = ModelPrototype->ReturnValues();
  revng_log(Log, "Updating return values");

  if (LLVMRetType->isIntOrPtrTy()) {
    revng_assert(ModelRetVals.size() == 1);
  } else if (LLVMRetType->isVoidTy()) {
    revng_assert(ModelRetVals.size() == 0);
  } else {
    const auto *RetValStruct = llvm::cast<llvm::StructType>(LLVMRetType);
    const auto &SubTypes = RetValStruct->subtypes();
    revng_assert(SubTypes.size() == ModelRetVals.size());

    const auto IsScalar = [](const llvm::Type *T) { return T->isIntOrPtrTy(); };
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
    auto &ModelReturnType = ModelRet.value().Type();
    if (canBeUpgraded(ModelReturnType)) {
      LayoutTypePtr Key{ LLVMRetVal, Index };
      if (auto NewTypeIt = TypeMap.find(Key); NewTypeIt != TypeMap.end()) {
        auto OldSize = ModelReturnType.size();
        // The type is associated to a LayoutTypeSystemPtr, hence we have to
        // add the pointer qualifier
        ModelReturnType = NewTypeIt->second.getPointerTo(Model.Architecture());
        revng_assert(ModelReturnType.size() == OldSize);
        revng_log(Log,
                  "Updated to "
                    << ModelReturnType.UnqualifiedType().get()->ID());
        Updated = true;
      }
    }
  }

  return Updated;
}

/// If the DLA recovered a more precise type than the already existing one
/// for any of the arguments of a CABIFunctionType, update the model
/// accordingly.
template<typename T>
// clang-format off
requires std::same_as<std::remove_const_t<T>, llvm::CallInst> or
         std::same_as<std::remove_const_t<T>, llvm::Function>
// clang-format on
static bool updateArgumentTypes(model::Binary &Model,
                                CABIFunctionType *ModelPrototype,
                                const T *CallOrFunction,
                                const TypeMapT &DLATypes) {
  bool Updated = false;
  auto &ModelArgs = ModelPrototype->Arguments();
  auto LLVMArgs = getArgs(CallOrFunction);

  uint64_t EffectiveLLVMArgSize = arg_size(CallOrFunction);
  revng_assert(ModelArgs.size() == EffectiveLLVMArgSize
               or (ModelArgs.size() == 1 and ModelArgs.begin()->Type().isVoid()
                   and EffectiveLLVMArgSize == 0));

  revng_log(Log, "Updating arguments");

  for (const auto &[LLVMArg, ModelArg] : llvm::zip_first(LLVMArgs, ModelArgs)) {
    const llvm::Value *LLVMVal = toLLVMValue(LLVMArg);
    revng_log(Log, "Updating argument " << LLVMVal->getNameOrAsOperand());

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have
    // no accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this
    // stage.
    if (canBeUpgraded(ModelArg.Type())) {
      LayoutTypePtr Key{ LLVMVal, LayoutTypePtr::fieldNumNone };
      if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {
        auto OldSize = *ModelArg.Type().size();
        // The type is associated to a LayoutTypeSystemPtr, hence we have to add
        // the pointer qualifier
        ModelArg.Type() = NewTypeIt->second.getPointerTo(Model.Architecture());
        revng_assert(*ModelArg.Type().size() == OldSize);
        revng_log(Log,
                  "Updated to "
                    << ModelArg.Type().UnqualifiedType().get()->ID());
        Updated = true;
      }
    }
  }

  return Updated;
}

/// If the DLA recovered a more precise type than the already existing one
/// for the return type of a CABIFunctionType, update the model accordingly.
static bool updateReturnType(model::Binary &Model,
                             CABIFunctionType *ModelPrototype,
                             const llvm::Value *LLVMRetVal,
                             const llvm::Type *LLVMRetType,
                             const TypeMapT &TypeMap) {
  auto &ModelReturnType = ModelPrototype->ReturnType();
  revng_log(Log, "Updating return value");

  // If the return type is void there's nothing to do.
  if (LLVMRetType->isVoidTy()) {
    revng_log(Log, "Is void");
    revng_assert(not ModelReturnType.UnqualifiedType().empty()
                 or ModelReturnType.isVoid());
    return false;
  }

  if (ModelReturnType.isScalar()) {
    if (!LLVMRetType->isIntOrPtrTy()) {
      revng_log(Log,
                "WARNING: model::CABIFunctionType returns a scalar type, the "
                "associated llvm::Function should return an integer or a "
                "pointer: "
                  << LLVMRetVal->getNameOrAsOperand());
      // If this happens we have an aggregate on LLVMIR and a scalar on
      // CABIFunction type. This should only happen in corner cases and we don't
      // have a general way to solve it properly right now, so we bail out and
      // never update that.
      return false;
    }

    revng_log(Log, "Is scalar: " << LLVMRetVal->getNameOrAsOperand());
    revng_assert(ModelReturnType.isScalar());

    // Don't update if the type is already fine-grained or if the DLA has
    // nothing to say.
    // The latter situation can happen e.g. with unused variables, that have
    // no accesses in the TypeSystem graph. These nodes are pruned away in the
    // middle-end, therefore there is no Type associated to them at this
    // stage.
    if (canBeUpgraded(ModelReturnType)) {
      LayoutTypePtr Key{ LLVMRetVal, LayoutTypePtr::fieldNumNone };
      if (auto NewTypeIt = TypeMap.find(Key); NewTypeIt != TypeMap.end()) {
        auto OldSize = *ModelReturnType.size();
        // The type is associated to a LayoutTypeSystemPtr, hence we have to
        // add the pointer qualifier
        ModelReturnType = NewTypeIt->second.getPointerTo(Model.Architecture());
        revng_assert(ModelReturnType.size() == OldSize);
        revng_log(Log,
                  "Updated to "
                    << ModelReturnType.UnqualifiedType().get()->ID());
        return true;
      }
    }
    return false;
  }

  // The case where CABIFunctionType's return type is scalar (meaning that
  // the corresponding IR also mentions a scalar) is handled above, as for
  // the aggregate case, we decided to just not do anything.
  //
  // The main reason behind such approach is the fact that usually non-scalar
  // return value types come from the places we should not override anyway,
  // i.e. debug information in the binary or the user.
  //
  // Also note, that big aggregate return value types are usually handled with
  // a shadow pointer argument, and so they are already scalars (pointers)
  // in the case where we deduce them manually.
  return false;
}

/// Update the prototype of a model::Function or of a call siet with the types
/// recovered by DLA.
template<typename T>
// clang-format off
requires std::same_as<std::remove_const_t<T>, llvm::CallInst> or
         std::same_as<std::remove_const_t<T>, llvm::Function>
// clang-format on
static bool updatePrototype(model::Binary &Model,
                            model::Type *Prototype,
                            const T *CallOrFunction,
                            const TypeMapT &TypeMap) {

  revng_assert(Prototype);
  bool Updated = false;

  revng_log(Log, "Updating func prototype");

  if (auto *RawPrototype = dyn_cast<RawFunctionType>(Prototype)) {
    Updated |= updateArgumentTypes(Model,
                                   RawPrototype,
                                   CallOrFunction,
                                   TypeMap);
    Updated |= updateReturnType(Model,
                                RawPrototype,
                                CallOrFunction,
                                getRetType(CallOrFunction),
                                TypeMap);
  } else if (auto *CABIPrototype = dyn_cast<CABIFunctionType>(Prototype)) {
    Updated |= updateArgumentTypes(Model,
                                   CABIPrototype,
                                   CallOrFunction,
                                   TypeMap);
    Updated |= updateReturnType(Model,
                                CABIPrototype,
                                CallOrFunction,
                                getRetType(CallOrFunction),
                                TypeMap);
  } else {
    revng_abort("Unsupported function type");
  }

  return Updated;
}

static void fillStructWithRecoveredDLAType(model::Binary &Model,
                                           model::Type *OriginalType,
                                           model::QualifiedType RecoveredType,
                                           uint64_t OriginalStructSize,
                                           uint64_t RecoveredStructSize) {
  revng_assert(isa<model::StructType>(OriginalType));
  revng_assert(RecoveredStructSize > 0
               and RecoveredStructSize < std::numeric_limits<int64_t>::max());

  auto *OriginalStructType = cast<model::StructType>(OriginalType);
  bool IsPointerOrArray = RecoveredType.isArray() || RecoveredType.isPointer();
  auto *RecoveredUnqualType = RecoveredType.UnqualifiedType().get();

  if (IsPointerOrArray or isa<model::PrimitiveType>(RecoveredUnqualType)
      or isa<model::EnumType>(RecoveredUnqualType)) {
    revng_assert(RecoveredStructSize <= OriginalStructSize);

    // If OriginalStructType is an empty struct, just add the new stack type
    // as the first field, otherwise leave it alone.
    if (OriginalStructType->Fields().empty()) {
      OriginalStructType->Fields()[0].Type() = RecoveredType;
      return;
    }

    auto FirstFieldIt = OriginalStructType->Fields().begin();
    model::StructField &First = *FirstFieldIt;
    // If there is already a field at offset 0, bail out
    if (First.Offset() == 0)
      return;

    // If the first original field is at an offset larger than 0, and the
    // RecoveredType fits, inject it as field at offset 0
    if (First.Offset() >= RecoveredStructSize)
      OriginalStructType->Fields()[0].Type() = RecoveredType;

    return;

  } else if (auto *NewS = dyn_cast<model::StructType>(RecoveredUnqualType)) {
    // If DLA recoverd a struct, whose size is too large, we have to shrink
    // it. For now the shrinking does not look deeply inside the types, only
    // at step one into the fields of the struct. We drop all the fields
    // that go over the OriginalStructSize.
    if (RecoveredStructSize > OriginalStructSize) {
      const auto IsTooLarge = [OriginalStructSize](const auto &Field) {
        return (Field.Offset() + *Field.Type().size()) > OriginalStructSize;
      };
      auto It = llvm::find_if(NewS->Fields(), IsTooLarge);
      auto End = NewS->Fields().end();
      NewS->Fields().erase(It, End);
      NewS->Size() = OriginalStructSize;
    }

    // Best case scenario, the recovered type struct size is less or equal than
    // the size of the original struct. In this case, just make sure we are not
    // introducing new overlapping fields with the original ones, if they exist.
    std::set<model::StructField *> CompatibleFields;
    auto OriginalFieldsIt = OriginalStructType->Fields().begin();
    auto OriginalFieldsEnd = OriginalStructType->Fields().end();
    auto NewFieldsIt = NewS->Fields().begin();
    auto NewFieldsEnd = NewS->Fields().end();

    while (NewFieldsIt != NewFieldsEnd) {
      // If we've reached the end of the original fields, all the remaining new
      // fields are compatible, because we've already filtered out all the
      // fields that start or end past the end of the original.
      if (OriginalFieldsIt == OriginalFieldsEnd) {
        for (model::StructField &Field :
             llvm::make_range(NewFieldsIt, NewFieldsEnd))
          CompatibleFields.insert(&Field);
        // Then we're done, because we've marked all the remaining new fields as
        // compatible.
        break;
      }

      uint64_t OriginalStart = OriginalFieldsIt->Offset();
      uint64_t OriginalEnd = OriginalFieldsIt->Offset()
                             + *OriginalFieldsIt->Type().size();

      uint64_t NewStart = NewFieldsIt->Offset();
      uint64_t NewEnd = NewFieldsIt->Offset() + *NewFieldsIt->Type().size();

      if (NewEnd <= OriginalStart) {
        CompatibleFields.insert(&*NewFieldsIt);
        ++NewFieldsIt;
        continue;
      }

      if (OriginalEnd <= NewStart) {
        ++OriginalFieldsIt;
        continue;
      }

      // If we reach this point, the new field is definitely overlapping, so we
      // have to skip to the next.
      ++NewFieldsIt;
    }

    // If all the fields are compatible, and the RecoveredType fits, try to
    // inject it as a single field of the OldStructType.
    bool AllNewAreCompatible = CompatibleFields.size() == NewS->Fields().size();
    if (AllNewAreCompatible) {

      // If the original struct was empty, the new one always fits.
      if (OriginalStructType->Fields().empty()) {
        OriginalStructType->Fields()[0].Type() = RecoveredType;
        return;
      }

      auto FirstFieldIt = OriginalStructType->Fields().begin();
      model::StructField &First = *FirstFieldIt;
      // If the first field of the original struct started after the new
      // recovered struct should end, we can always inject it.
      if (First.Offset() >= RecoveredStructSize) {
        OriginalStructType->Fields()[0].Type() = RecoveredType;
        return;
      }
    }

    for (model::StructField *NewField : CompatibleFields)
      OriginalStructType->Fields().insert(*NewField);

  } else if (auto *NewU = dyn_cast<model::UnionType>(RecoveredUnqualType)) {
    // If OriginalStructType is an not an empty struct, just leave it alone.
    if (not OriginalStructType->Fields().empty())
      return;

    // If DLA recoverd an union kind, whose size is too large, we have to
    // shrink it, like we did with a struct kind.
    auto FieldsRemaining = NewU->Fields().size();
    if (RecoveredStructSize > OriginalStructSize) {

      const auto IsTooLarge =
        [OriginalStructSize](const model::UnionField &Field) {
          return *Field.Type().size() > OriginalStructSize;
        };

      // First, detect the fields that are too large.
      llvm::SmallSet<size_t, 8> FieldsToDrop;
      auto It = NewU->Fields().begin();
      auto End = NewU->Fields().end();
      for (; It != End; ++It) {
        revng_assert(It->verify());
        revng_assert(It->Type().verify());
        if (IsTooLarge(*It))
          FieldsToDrop.insert(It->Index());
      }

      revng_assert(not FieldsToDrop.empty());

      // Count the fields that are left in the union if we remove those
      // selected to be dropped.
      FieldsRemaining = NewU->Fields().size() - FieldsToDrop.size();

      // If we need to drop at least one field but not all of them, we have
      // to renumber the remaining fields, otherwise the union will not
      // verify.
      if (FieldsRemaining) {

        // Drop the large fields
        for (auto &FieldNum : FieldsToDrop)
          NewU->Fields().erase(FieldNum);

        // Re-enumerate the remaining fields
        for (auto &Group : llvm::enumerate(NewU->Fields()))
          Group.value().Index() = Group.index();

        revng_assert(NewU->Fields().size() == FieldsRemaining);
        revng_assert(NewU->verify());
      }
    }

    // If there are fields left in the union, then inject the union in the
    // struct as field at offset 0.
    if (FieldsRemaining) {
      OriginalStructType->Fields()[0]
        .Type() = model::QualifiedType(Model.getTypePath(NewU), {});
    }
  } else {
    revng_abort();
  }
}

static bool updateStackFrameType(model::Function &ModelFunc,
                                 const llvm::Function &LLVMFunc,
                                 const TypeMapT &DLATypes,
                                 model::Binary &Model) {
  bool Updated = false;

  if (not ModelFunc.StackFrameType().isValid()
      or ModelFunc.StackFrameType().empty())
    return Updated;

  auto *OldModelStackFrameType = ModelFunc.StackFrameType().get();
  auto *OldStackFrameStruct = cast<model::StructType>(OldModelStackFrameType);
  if (not OldStackFrameStruct->Fields().empty())
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
    revng_log(Log, "Was " << OldModelStackFrameType->ID());

    LayoutTypePtr Key{ Call, LayoutTypePtr::fieldNumNone };

    if (auto NewTypeIt = DLATypes.find(Key); NewTypeIt != DLATypes.end()) {
      model::QualifiedType NewStackType = NewTypeIt->second;
      NewStackType = peelConstAndTypedefs(NewStackType);

      auto NewStackSize = *NewStackType.size();
      uint64_t OldStackSize = *OldStackFrameStruct->size();

      fillStructWithRecoveredDLAType(Model,
                                     OldModelStackFrameType,
                                     NewStackType,
                                     OldStackSize,
                                     NewStackSize);

      revng_log(Log, "Updated to " << ModelFunc.StackFrameType().get()->ID());

      revng_assert(isa<model::StructType>(ModelFunc.StackFrameType().get()));
      revng_assert(*ModelFunc.StackFrameType().get()->size() == OldStackSize);
      revng_assert(ModelFunc.StackFrameType().get()->verify());

      Updated = true;
    }
  }

  return Updated;
}

bool dla::updateFuncSignatures(const llvm::Module &M,
                               TupleTree<model::Binary> &Model,
                               const TypeMapT &TypeMap,
                               FunctionMetadataCache &Cache) {
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

    Type *ModelPrototype = ModelFunc->Prototype().get();
    revng_log(Log,
              "Updating prototype of function "
                << LLVMFunc.getNameOrAsOperand());
    Updated |= updatePrototype(*Model, ModelPrototype, &LLVMFunc, TypeMap);
    Updated |= updateStackFrameType(*ModelFunc, LLVMFunc, TypeMap, *Model);

    // Update prototypes associated to indirect calls, if any are found
    for (const auto &Inst : LLVMFunc)
      if (const auto *I = llvm::dyn_cast<llvm::CallInst>(&Inst)) {
        auto Prototype = Cache.getCallSitePrototype(*Model.get(), I, ModelFunc);
        if (not Prototype.empty()) {
          revng_log(Log,
                    "Updating prototype of indirect call "
                      << I->getNameOrAsOperand());
          Updated |= updatePrototype(*Model, Prototype.get(), I, TypeMap);
        }
      }
  }

  if (ModelLog.isEnabled())
    writeToFile(Model->toString(), "model-after-func-update.yaml");
  if (VerifyLog.isEnabled())
    revng_assert(Model->verify());

  return Updated;
}

bool dla::updateSegmentsTypes(const llvm::Module &M,
                              TupleTree<model::Binary> &Model,
                              const TypeMapT &TypeMap) {
  bool Updated = false;

  for (const auto &F : FunctionTags::SegmentRef.functions(&M)) {
    const auto &[StartAddress, VirtualSize] = extractSegmentKeyFromMetadata(F);
    auto Segment = Model->Segments().at({ StartAddress, VirtualSize });

    // We know that the Segment's type is of StructType.
    // It's empty, we'll fill it up.
    auto *UnqualSegment = Segment.Type().UnqualifiedType().get();
    revng_assert(isa<model::StructType>(UnqualSegment));
    auto *SegmentStruct = cast<model::StructType>(UnqualSegment);
    auto SegmentStructSize = *SegmentStruct->size();

    LayoutTypePtr Key{ &F, LayoutTypePtr::fieldNumNone };
    if (auto TypeIt = TypeMap.find(Key); TypeIt != TypeMap.end()) {
      // Let's examine the recovered type by DLA.
      model::QualifiedType RecoveredSegmentType = TypeIt->second;
      auto RecoveredSegmentTypeSize = *RecoveredSegmentType.size();

      fillStructWithRecoveredDLAType(*Model,
                                     UnqualSegment,
                                     RecoveredSegmentType,
                                     SegmentStructSize,
                                     RecoveredSegmentTypeSize);

      auto *NewUnqualType = Segment.Type().UnqualifiedType().get();
      revng_log(Log, "Updated to " << NewUnqualType->ID());
      revng_assert(isa<model::StructType>(NewUnqualType));
      revng_assert(*NewUnqualType->size() == SegmentStructSize);
      revng_assert(NewUnqualType->verify());

      Updated = true;
    }
  }

  return Updated;
}
