//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <optional>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/TypedefType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Instruction;

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using model::Binary;
using model::QualifiedType;

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

  const auto Layout = abi::FunctionType::Layout::make(*Prototype);

  const auto IsNonShadow = [](const abi::FunctionType::Layout::Argument &A) {
    using namespace abi::FunctionType::ArgumentKind;
    return A.Kind != ShadowPointerToAggregateReturnValue;
  };

  auto NumArgs = LLVMFunc.arg_size();
  size_t NumNonShadowArgs = llvm::count_if(Layout.Arguments, IsNonShadow);
  revng_assert(NumNonShadowArgs == NumArgs);
  auto NonShadowArgs = llvm::make_filter_range(Layout.Arguments, IsNonShadow);

  for (const auto &[ArgModelType, LLVMArg] :
       llvm::zip_first(NonShadowArgs, LLVMFunc.args())) {

    QualifiedType ArgQualifiedType = ArgModelType.Type;
    if (not PointersOnly or ArgQualifiedType.isPointer())
      TypeMap.insert({ &LLVMArg, std::move(ArgQualifiedType) });
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
          auto
            ByteSize = model::Architecture::getPointerSize(Model
                                                             .Architecture());
          auto BitWidth = 8 * ByteSize;
          auto *LLVMIntPtrType = llvm::IntegerType::getIntNTy(Operand
                                                                ->getContext(),
                                                              BitWidth);
          QualifiedType IntPtrType = llvmIntToModelType(LLVMIntPtrType, Model);
          TypeMap.insert({ Operand, IntPtrType });
        }
        rc_return true;
      }
    }
  } else if (isa<llvm::ConstantInt>(Operand)
             or isa<llvm::GlobalVariable>(Operand)) {

    model::QualifiedType Type = modelType(Operand, Model);
    if (not PointersOnly or Type.isPointer())
      TypeMap.insert({ Operand, Type });

    rc_return true;

  } else if (isa<llvm::PoisonValue>(Operand)
             or isa<llvm::UndefValue>(Operand)) {
    // poison and undef are always integers
    llvm::Type *OperandType = Operand->getType();
    revng_assert(OperandType->isIntOrPtrTy());
    auto *IntType = dyn_cast<llvm::IntegerType>(OperandType);
    if (not IntType) { // It's a pointer
      auto ByteSize = model::Architecture::getPointerSize(Model.Architecture());
      auto BitWidth = 8 * ByteSize;
      IntType = llvm::IntegerType::getIntNTy(Operand->getContext(), BitWidth);
    }
    auto ConstType = llvmIntToModelType(IntType, Model);
    revng_assert(not ConstType.isPointer());

    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly)
      TypeMap.insert({ Operand, ConstType });
    rc_return true;

  } else if (auto *NullPtr = dyn_cast<llvm::ConstantPointerNull>(Operand)) {
    if (not PointersOnly) {
      auto PtrSize = model::Architecture::getPointerSize(Model.Architecture());
      auto NullPointerType = model::QualifiedType{
        Model.getPrimitiveType(model::PrimitiveTypeKind::Generic, PtrSize),
        /*Qualifiers*/ {}
      };
      TypeMap.insert({ Operand, NullPointerType });
    }
    rc_return true;
  }

  rc_return false;
}

/// Reconstruct the return type(s) of a Call instruction from its
/// prototype, if it's an isolated function. For non-isolated functions,
/// special rules apply to recover the returned type.
static TypeVector getReturnTypes(const llvm::CallInst *Call,
                                 const model::Function *ParentFunc,
                                 const Binary &Model,
                                 const ModelTypesMap &TypeMap) {
  TypeVector ReturnTypes;

  if (Call->getType()->isVoidTy())
    return {};

  // Check if we already have strong model information for this call
  ReturnTypes = getStrongModelInfo(Call, Model);

  if (not ReturnTypes.empty())
    return ReturnTypes;

  auto *CalledFunc = Call->getCalledFunction();
  revng_assert(CalledFunc);

  if (FunctionTags::Parentheses.isTagOf(CalledFunc)
      or FunctionTags::Copy.isTagOf(CalledFunc)
      or FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {

    const llvm::Value *Arg = Call->getArgOperand(0);

    if (auto *ConstInt = dyn_cast<llvm::ConstantInt>(Arg);
        ConstInt and FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
      unsigned BitWidth = ConstInt->getType()->getIntegerBitWidth();
      unsigned ByteSize = std::max(1U, BitWidth / 8U);
      using model::PrimitiveTypeKind::Signed;
      auto SignedInt = model::QualifiedType(Model.getPrimitiveType(Signed,
                                                                   ByteSize),
                                            {});
      revng_assert(SignedInt.verify());
      ReturnTypes.push_back(std::move(SignedInt));
    } else {
      // Forward the type
      auto It = TypeMap.find(Arg);
      if (It != TypeMap.end())
        ReturnTypes.push_back(It->second);
    }

  } else if (FunctionTags::QEMU.isTagOf(CalledFunc)
             or FunctionTags::Helper.isTagOf(CalledFunc)
             or FunctionTags::Exceptional.isTagOf(CalledFunc)
             or CalledFunc->isIntrinsic()
             or FunctionTags::OpaqueCSVValue.isTagOf(CalledFunc)) {

    revng_assert(not CalledFunc->isTargetIntrinsic());

    llvm::Type *ReturnedType = Call->getType();

    if (ReturnedType->isSingleValueType()) {
      ReturnTypes.push_back(llvmIntToModelType(ReturnedType, Model));

    } else if (ReturnedType->isAggregateType()) {
      // For intrinsics and helpers returning aggregate types, we simply
      // return a list of all the subtypes, after transforming each in the
      // corresponding primitive QualifiedType
      for (llvm::Type *Subtype : ReturnedType->subtypes()) {
        ReturnTypes.push_back(llvmIntToModelType(Subtype, Model));
      }

    } else {
      revng_abort("Unknown value returned by non-isolated function");
    }
  } else if (FunctionTags::StringLiteral.isTagOf(CalledFunc)) {
    using model::PrimitiveTypeKind::Values::Unsigned;
    QualifiedType CharTy(Model.getPrimitiveType(Unsigned, 1), {});
    ReturnTypes.push_back(CharTy.getPointerTo(Model.Architecture()));
  } else if (FunctionTags::LiteralPrintDecorator.isTagOf(CalledFunc)) {
    const llvm::Value *Arg = Call->getArgOperand(0);
    ReturnTypes.push_back(llvmIntToModelType(Arg->getType(), Model));
  } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
    ReturnTypes.push_back(llvmIntToModelType(Call->getType(), Model));
  } else if (FunctionTags::BooleanNot.isTagOf(CalledFunc)) {
    auto IntType = llvm::IntegerType::getInt1Ty(CalledFunc->getContext());
    ReturnTypes.push_back(llvmIntToModelType(IntType, Model));
  } else {
    revng_abort("Unknown non-isolated function");
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

  if (ReturnedQualTypes.empty())
    return;

  llvm::Type *CallType = Call->getType();
  if (ReturnedQualTypes.size() == 1) {
    // If the function returns just one value, associate the computed
    // QualifiedType to the Call Instruction
    revng_assert(not CallType->isStructTy());

    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly or ReturnedQualTypes[0].isPointer()) {
      TypeMap.insert({ Call, ReturnedQualTypes[0] });
    }

  } else if (not CallType->isAggregateType()) {
    // If we reach this point, we have many types in ReturnedQualTypes, but the
    // Call on LLVM IR returns an integer.
    revng_assert(CallType->isIntegerTy());
    // In this case we cannot attach a rich type to the integer on LLVM IR, we
    // just have to fall back to a Generic PrimitiveType
    if (not PointersOnly) {
      const auto GenericKind = model::PrimitiveTypeKind::Generic;
      auto BitWidth = CallType->getIntegerBitWidth();
      revng_assert(BitWidth > 0 and not(BitWidth % 8));
      auto Generic = QualifiedType(Model.getPrimitiveType(GenericKind,
                                                          BitWidth / 8),
                                   {});
      TypeMap.insert({ Call, std::move(Generic) });
    }

  } else {
    // If we reach this point, we have many types in ReturnedQualTypes, and
    // the Call also returns a struct on LLVM IR

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
      for (const llvm::CallInst *ExtractValInst : ExtractedSet)
        // Skip if it's not a pointer and we are only interested in pointers
        if (not PointersOnly or QualType.isPointer())
          TypeMap.insert({ ExtractValInst, QualType });
    }
  }
}

static model::PrimitiveTypeKind::Values
getPrimitiveKind(const model::QualifiedType &QT) {
  revng_assert(QT.isPrimitive());
  model::QualifiedType Unwrapped = peelConstAndTypedefs(QT);
  revng_assert(Unwrapped.Qualifiers().empty());
  auto *Primitive = llvm::cast<model::PrimitiveType>(Unwrapped.UnqualifiedType()
                                                       .getConst());
  return Primitive->PrimitiveKind();
}

static model::PrimitiveTypeKind::Values
getCommonPrimitiveKind(model::PrimitiveTypeKind::Values A,
                       model::PrimitiveTypeKind::Values B) {
  if (A == B)
    return A;

  if (A == model::PrimitiveTypeKind::Generic
      or B == model::PrimitiveTypeKind::Generic)
    return model::PrimitiveTypeKind::Generic;

  // Here, neither A nor B are Generic

  // Given that A != B, and they're not generic, if either of them is Float, we
  // directly go to Generic.
  if (A == model::PrimitiveTypeKind::Float
      or B == model::PrimitiveTypeKind::Float)
    return model::PrimitiveTypeKind::Generic;

  // Here neither A nor B is Generic nor Float

  // If either is PointerOrNumber, we go to PointerOrNumber.
  if (A == model::PrimitiveTypeKind::PointerOrNumber
      or B == model::PrimitiveTypeKind::PointerOrNumber)
    return model::PrimitiveTypeKind::PointerOrNumber;

  // Here neither A nor B is Generic, Float, nor PointerOrNumber
  // Here A and B can only be Number, Signed or Unsigned.
  // Given that they are different, we always go to Number.
  return model::PrimitiveTypeKind::Number;
}

static model::QualifiedType
getEnumUnderlyingType(const model::QualifiedType &QT) {
  revng_assert(QT.is(model::TypeKind::EnumType));
  model::QualifiedType Unwrapped = peelConstAndTypedefs(QT);
  revng_assert(Unwrapped.Qualifiers().empty());
  auto *Enum = llvm::cast<model::EnumType>(Unwrapped.UnqualifiedType()
                                             .getConst());
  return Enum->UnderlyingType();
}

static std::optional<model::QualifiedType>
getCommonScalarType(const model::QualifiedType &A,
                    const model::QualifiedType &B,
                    const model::Binary &Model) {

  using model::PrimitiveTypeKind::Values::Float;
  using model::PrimitiveTypeKind::Values::Generic;
  using model::PrimitiveTypeKind::Values::PointerOrNumber;

  revng_assert(A.isScalar());
  revng_assert(B.isScalar());

  if (A == B)
    return A;

  revng_assert(A.isPrimitive() or A.isPointer()
               or A.is(model::TypeKind::EnumType));
  revng_assert(B.isPrimitive() or B.isPointer()
               or B.is(model::TypeKind::EnumType));

  revng_assert(A.size() == B.size());
  uint64_t Size = A.size().value();

  if (A.isPrimitive() and B.isPrimitive()) {
    model::PrimitiveTypeKind::Values AKind = getPrimitiveKind(A);
    model::PrimitiveTypeKind::Values BKind = getPrimitiveKind(B);
    model::PrimitiveTypeKind::Values CommonKind = getCommonPrimitiveKind(AKind,
                                                                         BKind);
    return model::QualifiedType(Model.getPrimitiveType(CommonKind, Size), {});
  }

  if (A.isPrimitive() or B.isPrimitive()) {

    const model::QualifiedType &Primitive = A.isPrimitive() ? A : B;
    model::PrimitiveTypeKind::Values
      PrimitiveKind = getPrimitiveKind(Primitive);

    const model::QualifiedType &Other = A.isPrimitive() ? B : A;

    if (Other.isPointer()) {

      if (PrimitiveKind == Generic)
        return Other;

      if (PrimitiveKind == Float)
        return model::QualifiedType(Model.getPrimitiveType(Generic, Size), {});

      return model::QualifiedType(Model.getPrimitiveType(PointerOrNumber, Size),
                                  {});

    } else if (Other.is(model::TypeKind::EnumType)) {

      model::PrimitiveTypeKind::Values
        OtherKind = getPrimitiveKind(getEnumUnderlyingType(Other));

      model::PrimitiveTypeKind::Values
        CommonKind = getCommonPrimitiveKind(PrimitiveKind, OtherKind);

      return model::QualifiedType(Model.getPrimitiveType(CommonKind, Size), {});

    } else {
      revng_abort();
    }
  }

  // Here neither A nor B are primitive. They are either enums or pointers.
  // If one is a pointer and the other is an enum, we can't find a common type.
  if (A.isPointer() and B.is(model::TypeKind::EnumType))
    return std::nullopt;
  if (B.isPointer() and A.is(model::TypeKind::EnumType))
    return std::nullopt;

  if (A.is(model::TypeKind::EnumType) and B.is(model::TypeKind::EnumType)) {
    // Make the common integer among the underlying types
    model::PrimitiveTypeKind::Values
      AKind = getPrimitiveKind(getEnumUnderlyingType(A));
    model::PrimitiveTypeKind::Values
      BKind = getPrimitiveKind(getEnumUnderlyingType(B));
    model::PrimitiveTypeKind::Values CommonKind = getCommonPrimitiveKind(AKind,
                                                                         BKind);
    return model::QualifiedType(Model.getPrimitiveType(CommonKind, Size), {});
  }

  if (A.isPointer() and B.isPointer()) {
    // Make a pointerornumber of the proper size (or could we do a void *)
    return model::QualifiedType(Model.getPrimitiveType(PointerOrNumber, Size),
                                {});
  }

  // This should be unreachable, but we return a nullopt, to fail gracefully
  return std::nullopt;
}

static llvm::SmallPtrSet<const llvm::Value *, 8>
getTransitivePHIIncomings(const llvm::PHINode *PHI) {
  llvm::SmallPtrSet<const llvm::Value *, 8> NonPHIIncomings;

  llvm::SmallPtrSet<const llvm::PHINode *, 8> VisitedPHIs = { PHI };
  llvm::SmallVector<const llvm::PHINode *> WorkList = { PHI };

  do {
    const llvm::PHINode *Current = WorkList.back();
    WorkList.pop_back();
    for (const llvm::Value *Incoming : Current->incoming_values()) {
      if (const auto *IncomingPHI = dyn_cast<llvm::PHINode>(Incoming)) {
        if (bool New = VisitedPHIs.insert(IncomingPHI).second)
          WorkList.push_back(IncomingPHI);
      } else {
        NonPHIIncomings.insert(Incoming);
      }
    }
  } while (not WorkList.empty());

  return NonPHIIncomings;
}

static RecursiveCoroutine<std::optional<QualifiedType>>
initModelTypesImpl(const llvm::Instruction &I,
                   const llvm::Function &F,
                   const model::Function *ModelF,
                   const Binary &Model,
                   bool PointersOnly,
                   ModelTypesMap &TypeMap,
                   llvm::SmallPtrSet<const llvm::PHINode *, 8>
                     VisitedPHIs = {}) {

  const auto *InstType = I.getType();

  // Ignore operands of some custom opcodes
  if (not isCallTo(&I, "revng_call_stack_arguments")) {
    // Visit operands, in case they are constants, globals or constexprs
    for (const llvm::Use &Op : I.operands()) {

      if (auto *Call = getCallToIsolatedFunction(&I);
          Call and Call->isCallee(&Op)) {
        // Isolated functions have their prototype in the model
        //
        // If it's a direct call to an isolated function we know the type of
        // the function, which affects the type of the
        auto *Called = Call->getCalledOperand();
        if (auto *CalledFunction = dyn_cast<llvm::Function>(Called)) {
          auto Prototype = getCallSitePrototype(Model, Call);
          revng_assert(Prototype.isValid() and not Prototype.empty());
          TypeMap.insert({ CalledFunction, createPointerTo(Prototype, Model) });
          continue;
        }
      }
      addOperandType(Op, Model, TypeMap, PointersOnly);
    }
  }

  // Insert void types for consistency
  if (InstType->isVoidTy()) {
    using model::PrimitiveTypeKind::Values::Void;
    QualifiedType VoidTy(Model.getPrimitiveType(Void, 0), {});
    TypeMap.insert({ &I, VoidTy });
    rc_return VoidTy;
  }
  // Function calls in the IR might correspond to real function calls in
  // the binary or to special intrinsics used by the backend, so they need
  // to be handled separately
  if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
    handleCallInstruction(Call, ModelF, Model, TypeMap, PointersOnly);
    auto CallTypeIt = TypeMap.find(Call);
    std::optional<QualifiedType> CallType = std::nullopt;
    if (CallTypeIt != TypeMap.end())
      CallType = CallTypeIt->second;
    rc_return CallType;
  }

  // Only Call instructions can return aggregates
  revng_assert(not InstType->isAggregateType());

  // All ExtractValues should have been converted to OpaqueExtractValue
  revng_assert(not isa<llvm::ExtractValueInst>(&I));

  std::optional<QualifiedType> Type = std::nullopt;

  switch (I.getOpcode()) {

  case Instruction::Load: {
    auto *Load = dyn_cast<llvm::LoadInst>(&I);

    auto It = TypeMap.find(Load->getPointerOperand());
    if (It == TypeMap.end())
      rc_return std::nullopt;

    const auto &PtrOperandType = It->second;

    // If the pointer operand is a pointer in the model, we can exploit
    // this information to assign a model type to the loaded value. Note
    // that this makes sense only if the pointee is itself a pointer or a
    // scalar value: if we find a load of N bits from a struct pointer, we
    // don't know if we are loading the entire struct or only some of its
    // fields.
    // TODO: inspect the model to understand if we are loading the first
    // field.
    if (PtrOperandType.isPointer()) {
      model::QualifiedType Pointee = dropPointer(PtrOperandType);

      if (areMemOpCompatible(Pointee, *Load->getType(), Model))
        Type = Pointee;
    }

    // If it's not a pointer or a scalar of the right size, just
    // fallback to the LLVM type

  } break;

  case Instruction::Alloca: {
    // TODO: eventually AllocaInst will be replaced by calls to
    // revng_local_variable with a type annotation
    llvm::Type *BaseType = cast<llvm::AllocaInst>(&I)->getAllocatedType();
    revng_assert(BaseType->isSingleValueType());
    const model::Architecture::Values &Architecture = Model.Architecture();
    Type = llvmIntToModelType(BaseType, Model).getPointerTo(Architecture);
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

  // Handle zext from i1 to i8
  case Instruction::ZExt: {
    auto *ZExt = dyn_cast<llvm::ZExtInst>(&I);

    auto IsBoolZext = ZExt->getSrcTy()->getScalarSizeInBits() == 1
                      and ZExt->getDestTy()->getScalarSizeInBits() == 8;

    if (not PointersOnly and IsBoolZext) {
      const llvm::Value *Operand = I.getOperand(0);

      // Forward the type if there is one
      auto It = TypeMap.find(Operand);
      if (It != TypeMap.end())
        Type = It->second;
    }

  } break;

  // Handle trunc from i8 to i1
  case Instruction::Trunc: {
    auto *Trunc = dyn_cast<llvm::TruncInst>(&I);

    auto IsBoolTrunc = Trunc->getSrcTy()->getScalarSizeInBits() == 8
                       and Trunc->getDestTy()->getScalarSizeInBits() == 1;

    if (not PointersOnly and IsBoolTrunc) {
      const llvm::Value *Operand = I.getOperand(0);

      // Forward the type if there is one
      auto It = TypeMap.find(Operand);
      if (It != TypeMap.end())
        Type = It->second;
    }

  } break;

  case Instruction::BitCast:
  case Instruction::Freeze:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt: {
    // Forward the type if there is one
    auto It = TypeMap.find(I.getOperand(0));
    if (It != TypeMap.end()) {
      const QualifiedType &OperandType = It->second;
      if (OperandType.isPointer()) {
        Type = OperandType;
      } else if (not PointersOnly) {
        auto ByteSize = model::Architecture::getPointerSize(Model
                                                              .Architecture());
        auto BitWidth = 8 * ByteSize;
        auto *LLVMIntPtrType = llvm::IntegerType::getIntNTy(I.getContext(),
                                                            BitWidth);
        Type = llvmIntToModelType(LLVMIntPtrType, Model);
        revng_assert(ByteSize == Type->size());
      }
    }
  } break;

  case Instruction::PHI: {
    auto *PHI = cast<llvm::PHINode>(&I);
    bool New = VisitedPHIs.insert(PHI).second;
    if (New) {
      llvm::SmallPtrSet<const llvm::Value *, 8>
        NonPHIIncomings = getTransitivePHIIncomings(PHI);

      for (const llvm::Value *Incoming : NonPHIIncomings) {
        std::optional<QualifiedType> IncomingType = std::nullopt;
        auto IncomingTypeIt = TypeMap.find(Incoming);
        if (IncomingTypeIt != TypeMap.end()) {
          IncomingType = IncomingTypeIt->second;
        } else {
          if (auto
                *IncomingInstruction = dyn_cast<llvm::Instruction>(Incoming)) {
            IncomingType = rc_recur initModelTypesImpl(*IncomingInstruction,
                                                       F,
                                                       ModelF,
                                                       Model,
                                                       PointersOnly,
                                                       TypeMap,
                                                       VisitedPHIs);
          }
        }
        if (not IncomingType)
          continue;

        if (not Type) {
          Type = IncomingType;
        } else {
          std::optional<model::QualifiedType>
            CommonType = getCommonScalarType(*Type, *IncomingType, Model);
          if (CommonType.has_value())
            Type = CommonType.value();
          else
            Type = llvmIntToModelType(PHI->getType(), Model);
        }
      }
    }
  } break;

  default:
    break;
  }

  rc_return Type;
}

static RecursiveCoroutine<ModelTypesMap>
initModelTypesImpl(const llvm::Function &F,
                   const model::Function *ModelF,
                   const Binary &Model,
                   bool PointersOnly,
                   llvm::SmallPtrSet<const llvm::PHINode *, 8>
                     VisitedPHIs = {}) {

  ModelTypesMap TypeMap;

  const model::Type *Prototype = ModelF->prototype(Model).getConst();
  revng_assert(Prototype);

  addArgumentsTypes(F, Prototype, Model, TypeMap, PointersOnly);

  for (const BasicBlock *BB : RPOT<const llvm::Function *>(&F)) {
    for (const Instruction &I : *BB) {
      std::optional<QualifiedType> Type = initModelTypesImpl(I,
                                                             F,
                                                             ModelF,
                                                             Model,
                                                             PointersOnly,
                                                             TypeMap,
                                                             VisitedPHIs);
      if (PointersOnly) {
        // Skip if it's not a pointer and we are only interested in pointers
        if (Type and Type->isPointer())
          TypeMap.insert({ &I, *Type });

      } else {
        // As a fallback, use the LLVM type to build the QualifiedType
        if (not Type and I.getType()->isIntOrPtrTy())
          Type = llvmIntToModelType(I.getType(), Model);

        if (Type)
          TypeMap.insert({ &I, *Type });
      }
    }
  }

  rc_return TypeMap;
}

ModelTypesMap initModelTypes(const llvm::Function &F,
                             const model::Function *ModelF,
                             const Binary &Model,
                             bool PointersOnly) {
  return initModelTypesImpl(F, ModelF, Model, PointersOnly);
}
