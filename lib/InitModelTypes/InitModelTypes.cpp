//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/TypedefType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Instruction;
using llvm::StringRef;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using model::Binary;
using model::CABIFunctionType;
using model::QualifiedType;
using model::RawFunctionType;

template<typename T>
using RPOT = llvm::ReversePostOrderTraversal<T>;

using TypeVector = llvm::SmallVector<QualifiedType, 8>;
using ModelTypesMap = std::map<const llvm::Value *, const model::QualifiedType>;

/// Map each llvm::Argument of the given llvm::Function to its
/// QualifiedType in the model.
static void addArgumentsTypes(const llvm::Function &LLVMFunc,
                              const model::Type *Prototype,
                              const Binary &Model,
                              ModelTypesMap &TypeMap,
                              bool PointersOnly) {

  if (auto *RawPrototype = dyn_cast<model::RawFunctionType>(Prototype)) {
    const auto &ModelArgs = RawPrototype->Arguments;
    const auto &LLVMArgs = LLVMFunc.args();

    // Assign each argument in the model prototype to the corresponding LLVM
    // argument
    auto ModelArgsIt = ModelArgs.begin();
    auto LLVMArgsIt = LLVMArgs.begin();
    while (ModelArgsIt != ModelArgs.end()) {
      revng_assert(LLVMArgsIt != LLVMArgs.end());

      // Skip if it's not a pointer and we are only interested in pointers
      if (not PointersOnly or ModelArgsIt->Type.isPointer()) {
        TypeMap.insert({ LLVMArgsIt, ModelArgsIt->Type });
      }
      ++LLVMArgsIt;
      ++ModelArgsIt;
    }

    QualifiedType StackArgs = RawPrototype->StackArgumentsType;
    // If there is still an argument left, it's a pointer to the stack arguments
    if (StackArgs.UnqualifiedType.isValid()) {
      revng_assert(LLVMArgsIt != LLVMArgs.end());
      // It's a pointer by definition, we don't need to check the `PointersOnly`
      // flag
      addPointerQualifier(StackArgs, Model);
      TypeMap.insert({ LLVMArgsIt, StackArgs });

      ++LLVMArgsIt;
    }

    // There should be no remaining arguments to visit
    revng_assert(LLVMArgsIt == LLVMArgs.end());

  } else {
    // TODO: handle CABIFunctionTypes
    revng_abort("CABIFunctionTypes are not supported yet.");
  }
}

/// Create a QualifiedType for unvisited operands, i.e. constants,
/// globals and constexprs.
/// \return true if a new token has been generated for the operand
static RecursiveCoroutine<bool> addOperandType(const llvm::Value *Operand,
                                               const Binary &Model,
                                               ModelTypesMap &TypeMap,
                                               bool PointersOnly) {

  // For ConstExprs, check their OpCode
  if (auto *Expr = dyn_cast<llvm::ConstantExpr>(Operand)) {
    // A constant expression might have its own uninitialized constant operands
    for (const llvm::Value *Op : Expr->operand_values())
      rc_recur addOperandType(Op, Model, TypeMap, PointersOnly);

    if (Expr->getOpcode() == Instruction::IntToPtr) {
      auto It = TypeMap.find(Expr->getOperand(0));
      if (It != TypeMap.end()) {
        const QualifiedType &OperandType = It->second;

        if (OperandType.isPointer()) {
          // If the operand has already a pointer qualified type, forward it
          TypeMap.insert({ Operand, OperandType });

        } else if (not PointersOnly) {
          // Fallback to the LLVM type
          auto ConstType = llvmIntToModelType(Operand->getType(), Model);
          TypeMap.insert({ Operand, ConstType });
        }
        rc_return true;
      }
    }
  } else if (isa<llvm::ConstantInt>(Operand)
             or isa<llvm::GlobalVariable>(Operand)) {

    // For constants and globals, fallback to the LLVM type
    revng_assert(not Operand->getType()->isVoidTy());
    auto ConstType = llvmIntToModelType(Operand->getType(), Model);
    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly or ConstType.isPointer()) {
      TypeMap.insert({ Operand, ConstType });
    }
    rc_return true;

  } else if (isa<llvm::ConstantPointerNull>(Operand)) {
    if (not PointersOnly)
      TypeMap.insert({ Operand, {} });
    rc_return true;
  }

  rc_return false;
}

/// Return the \a Idx -th Type contained in \a QT.
/// \note QT must be an array, struct or union.
/// TODO: this could be avoided if we the GEPped type was embedded in the
/// ModelGEP call
static QualifiedType traverseModelGEP(const QualifiedType &QT, uint64_t Idx) {
  QualifiedType ReturnedQT = QT;
  auto *UnqualType = ReturnedQT.UnqualifiedType.getConst();

  // Peel Typedefs transparently
  while (auto *Typedef = dyn_cast<model::TypedefType>(UnqualType)) {
    const model::QualifiedType &Underlying = Typedef->UnderlyingType;
    // Copy Qualifiers
    for (const model::Qualifier &Q : Underlying.Qualifiers)
      ReturnedQT.Qualifiers.push_back(Q);
    // Visit Underlying type
    ReturnedQT.UnqualifiedType = Underlying.UnqualifiedType;
    UnqualType = ReturnedQT.UnqualifiedType.getConst();
  }

  // Remove Qualifiers from the right until we find an array qualifier
  auto It = ReturnedQT.Qualifiers.end();
  while (It != ReturnedQT.Qualifiers.begin()) {
    --It;

    if (model::Qualifier::isArray(*It)) {
      // If we are traversing an array, the Idx-th element has the same type of
      // the array, except for the array Qualifier
      ReturnedQT.Qualifiers.erase(It);
      return ReturnedQT;
    }

    ReturnedQT.Qualifiers.erase(It);
  }

  // If we arrived here, there are no qualifiers left to traverse
  revng_assert(It == ReturnedQT.Qualifiers.end());

  // Traverse the UnqualifiedType
  if (auto *Struct = dyn_cast<model::StructType>(UnqualType)) {
    ReturnedQT = Struct->Fields.at(Idx).Type;

  } else if (auto *Union = dyn_cast<model::UnionType>(UnqualType)) {
    ReturnedQT = Union->Fields.at(Idx).Type;

  } else {
    revng_abort("Unexpected ModelGEP type found: ");
    UnqualType->dump();
  }

  return ReturnedQT;
}

/// Reconstruct the return type(s) of a Call instruction from its
/// prototype, if it's an isolated function. For non-isolated functions,
/// special rules apply to recover the returned type.
static TypeVector getReturnTypes(const llvm::CallInst *Call,
                                 const model::Function *ParentFunc,
                                 const Binary &Model,
                                 ModelTypesMap &TypeMap) {
  TypeVector ReturnTypes;

  if (FunctionTags::CallToLifted.isTagOf(Call)) {
    // Retrieve the function prototype
    auto Prototype = getCallSitePrototype(Model, Call, ParentFunc);
    revng_assert(Prototype.isValid());
    auto PrototypePath = Prototype.get();

    // Collect returned type(s)
    if (auto *CPrototype = dyn_cast<CABIFunctionType>(PrototypePath)) {
      ReturnTypes.push_back(CPrototype->ReturnType);

    } else if (auto *RawPrototype = dyn_cast<RawFunctionType>(PrototypePath)) {
      for (const auto &RetVal : RawPrototype->ReturnValues)
        ReturnTypes.push_back(RetVal.Type);
    }

  } else {
    // Non-lifted functions do not have a Prototype in the model, but we can
    // infer their returned type(s) in other ways
    auto *CalledFunc = Call->getCalledFunction();
    const auto &FuncName = CalledFunc->getName();

    if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
      revng_assert(Call->getNumArgOperands() >= 2);
      // ModelGEPs contain a string with the serialization of the pointer's
      // base QualifiedType as a first argument
      StringRef FirstOp = extractFromConstantStringPtr(Call->getArgOperand(0));
      QualifiedType GEPpedType = parseQualifiedType(FirstOp, Model);

      // Second argument is the base llvm::Value
      // Further arguments are used to traverse the model
      auto CurArg = Call->arg_begin() + 2;
      for (; CurArg != Call->arg_end(); ++CurArg) {
        uint64_t Idx = 0;

        if (auto *ArgAsInt = dyn_cast<llvm::ConstantInt>(CurArg->get()))
          Idx = ArgAsInt->getValue().getLimitedValue();

        GEPpedType = traverseModelGEP(GEPpedType, Idx);
      }

      ReturnTypes.push_back(GEPpedType);

    } else if (FunctionTags::AddressOf.isTagOf(CalledFunc)) {
      // AddressOf contains a string with the serialization of the pointed type
      // as a first argument
      StringRef FirstOp = extractFromConstantStringPtr(Call->getArgOperand(0));
      QualifiedType PointedType = parseQualifiedType(FirstOp, Model);

      // Since we are taking the address, the final type will be a pointer to
      // the base type
      addPointerQualifier(PointedType, Model);
      ReturnTypes.push_back(PointedType);

    } else if (FunctionTags::AssignmentMarker.isTagOf(CalledFunc)) {
      const llvm::Value *Arg = Call->getArgOperand(0);

      // Structs are handled on their own
      if (Arg->getType()->isStructTy())
        return {};

      // AssignmentMarker is transparent
      auto It = TypeMap.find(Arg);
      if (It != TypeMap.end()) {
        ReturnTypes.push_back(It->second);
      }

    } else if (FunctionTags::StructInitializer.isTagOf(CalledFunc)) {
      // Struct initializers are only used to pack together return values of
      // RawFunctionTypes that return multiple values, therefore they have the
      // same type as the parent function's return type
      revng_assert(Call->getFunction()->getReturnType() == Call->getType());
      auto *RawPrototype = cast<RawFunctionType>(ParentFunc->Prototype.get());

      for (const auto &RetVal : RawPrototype->ReturnValues)
        ReturnTypes.push_back(RetVal.Type);

    } else if (FuncName.startswith("revng_stack_frame")) {
      // Retrieve the stack frame type
      auto &StackType = ParentFunc->StackFrameType;
      revng_assert(StackType.get());

      ReturnTypes.push_back(createPointerTo(StackType, Model));

    } else if (FuncName.startswith("revng_call_stack_arguments")) {
      // The prototype attached to this callsite represents the prototype of
      // the function that needs the stack arguments returned by this call
      auto Prototype = getCallSitePrototype(Model, Call, ParentFunc);
      revng_assert(Prototype.isValid());

      // Only RawFunctionTypes have explicit stack arguments
      auto *RawPrototype = cast<model::RawFunctionType>(Prototype.get());
      QualifiedType StackArgsType = RawPrototype->StackArgumentsType;
      addPointerQualifier(StackArgsType, Model);

      ReturnTypes.push_back(StackArgsType);

    } else if (FuncName.startswith("revng_init_local_sp")) {
      using model::PrimitiveTypeKind::Unsigned;

      // TODO: For now we use uint8_t* as the type returned by
      // revng_init_local_sp, but perhaps we could choose a more appropriate
      // type to represent SP
      QualifiedType StackPtrQT;
      StackPtrQT.UnqualifiedType = Model.getPrimitiveType(Unsigned, 1);
      addPointerQualifier(StackPtrQT, Model);

      ReturnTypes.push_back(StackPtrQT);

    } else if (FunctionTags::QEMU.isTagOf(CalledFunc)
               or FunctionTags::Helper.isTagOf(CalledFunc)
               or FuncName.startswith("llvm.")
               or FuncName.startswith("init_")) {

      llvm::Type *ReturnedType = Call->getType();
      if (ReturnedType->isVoidTy())
        return {};

      if (ReturnedType->isSingleValueType()) {
        ReturnTypes.push_back(llvmIntToModelType(ReturnedType, Model));

      } else if (ReturnedType->isAggregateType()) {
        // For intrinsics and helpers returning aggregate types, we simply
        // return a lit of all the subtypes, after transforming each in the
        // corresponding primitive QualifiedType
        for (llvm::Type *Subtype : ReturnedType->subtypes()) {
          ReturnTypes.push_back(llvmIntToModelType(Subtype, Model));
        }

      } else {
        revng_abort("Unknown value returned by non-isolated function");
      }

    } else {
      revng_abort("Unknown non-isolated function");
    }
  }

  return ReturnTypes;
}

/// Given a call instruction, to either an isolated or a non-isolated
/// function, assign to it its return type. If the call returns more than
/// one type, infect the uses of the returned value with those types.
static void handleCallInstruction(const llvm::CallInst *Call,
                                  const model::Function *ParentFunc,
                                  const Binary &Model,
                                  ModelTypesMap &TypeMap,
                                  bool PointersOnly) {

  TypeVector ReturnedQualTypes = getReturnTypes(Call,
                                                ParentFunc,
                                                Model,
                                                TypeMap);

  if (ReturnedQualTypes.size() == 0)
    return;

  if (ReturnedQualTypes.size() == 1) {
    // If the function returns just one value, associate the computed
    // QualifiedType to the Call Instruction
    revng_assert(Call->getType()->isSingleValueType());

    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly or ReturnedQualTypes[0].isPointer()) {
      TypeMap.insert({ Call, ReturnedQualTypes[0] });
    }

  } else {
    revng_assert(Call->getType()->isAggregateType());

    // Functions that return aggregate types have more than one return type.
    // In this case, we cannot assign all the returned types to the returned
    // llvm::Value. Hence, we collect the returned types in a vector and
    // assign them to the values extracted from the returned struct.
    const auto ExtractedValues = getExtractedValuesFromInstruction(Call);
    revng_assert(ReturnedQualTypes.size() == ExtractedValues.size());

    for (const auto &ZippedRetVals : zip(ReturnedQualTypes, ExtractedValues)) {
      const auto &[QualType, ExtractedSet] = ZippedRetVals;
      revng_assert(QualType.isScalar());

      // Each extractedSet contains the set of instructions that extract the
      // same value from the struct
      for (const llvm::ExtractValueInst *ExtractValInst : ExtractedSet)
        // Skip if it's not a pointer and we are only interested in pointers
        if (not PointersOnly or QualType.isPointer())
          TypeMap.insert({ ExtractValInst, QualType });
    }
  }
}

ModelTypesMap initModelTypes(const llvm::Function &F,
                             const model::Function *ModelF,
                             const Binary &Model,
                             bool PointersOnly) {
  ModelTypesMap TypeMap;

  const model::Type *Prototype = ModelF->Prototype.getConst();
  revng_assert(Prototype);

  addArgumentsTypes(F, Prototype, Model, TypeMap, PointersOnly);

  for (const BasicBlock *BB : RPOT<const llvm::Function *>(&F)) {
    for (const Instruction &I : *BB) {
      const auto *InstType = I.getType();

      // Visit operands, in case they are constants, globals or constexprs
      for (const llvm::Value *Op : I.operand_values())
        addOperandType(Op, Model, TypeMap, PointersOnly);

      // Insert void types for consistency, although they
      if (InstType->isVoidTy()) {
        using model::PrimitiveTypeKind::Values::Void;
        QualifiedType VoidTy(Model.getPrimitiveType(Void, 0), {});
        TypeMap.insert({ &I, VoidTy });
        continue;
      }
      // Function calls in the IR might correspond to real function calls in
      // the binary or to special intrinsics used by the backend, so they need
      // to be handled separately
      if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
        handleCallInstruction(Call, ModelF, Model, TypeMap, PointersOnly);
        continue;
      }

      // Only Call instructions can return aggregates
      revng_assert(not InstType->isAggregateType());

      // All InsertValues and ExtractValues should have been assigned when
      // handling Call instructions that return an aggregate
      if (isa<llvm::ExtractValueInst>(&I)) {
        if (not PointersOnly)
          revng_assert(TypeMap.contains(&I));
        continue;
      }

      llvm::Optional<QualifiedType> Type;

      switch (I.getOpcode()) {

      case Instruction::Load: {
        auto *Load = dyn_cast<llvm::LoadInst>(&I);

        auto It = TypeMap.find(Load->getPointerOperand());
        if (It == TypeMap.end())
          continue;

        const auto &PtrOperandType = It->second;

        // If the pointer operand is a pointer in the model, we can exploit
        // this information to assign a model type to the loaded value. Note
        // that this makes sense only if the pointee is itself a pointer or a
        // scalar value: if we find a lod of N bits from a struct pointer, we
        // don't know if we are loading the entire struct or only some of its
        // fields.
        // TODO: inspect the model to understand if we are loading the first
        // field.
        if (PtrOperandType.isPointer()) {
          model::QualifiedType Pointee = dropPointer(PtrOperandType);
          if (Pointee.isPointer() or Pointee.isScalar()) {
            Type = Pointee;
          }
        }

      } break;

      case Instruction::Alloca: {
        // TODO: eventually AllocaInst will be replaced by calls to
        // revng_local_variable with a type annotation
        llvm::PointerType *PtrType = llvm::cast<llvm::PointerType>(I.getType());
        llvm::Type *BaseType = PtrType->getElementType();
        revng_assert(BaseType->isSingleValueType());
        Type = llvmIntToModelType(BaseType, Model);
        addPointerQualifier(*Type, Model);

      } break;

      case Instruction::Select: {
        auto *Select = dyn_cast<llvm::SelectInst>(&I);
        const auto &Op1Entry = TypeMap.find(Select->getOperand(1));
        const auto &Op2Entry = TypeMap.find(Select->getOperand(2));

        // If the two selected values have the same type, assign that type to
        // the result
        if (Op1Entry != TypeMap.end() and Op2Entry != TypeMap.end()
            and Op1Entry->second == Op2Entry->second)
          Type = Op1Entry->second;

      } break;

      case Instruction::IntToPtr: {
        const llvm::IntToPtrInst *IntToPtr = dyn_cast<llvm::IntToPtrInst>(&I);
        const llvm::Value *Operand = IntToPtr->getOperand(0);

        auto It = TypeMap.find(Operand);
        if (It == TypeMap.end())
          continue;

        const QualifiedType &OperandType = It->second;

        // If the operand has already a pointer qualified type, forward it
        if (OperandType.isPointer()) {
          Type = OperandType;
        }

      } break;

      default:
        break;
      }

      // As a fallback, use the LLVM type to build the QualifiedType
      if (not Type)
        Type = llvmIntToModelType(InstType, Model);

      // Skip if it's not a pointer and we are only interested in pointers
      if (not PointersOnly or Type->isPointer())
        TypeMap.insert({ &I, *Type });
    }
  }

  return TypeMap;
}
