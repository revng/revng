//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <optional>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/DefinedType.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/DecompilationHelpers.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"

using llvm::BasicBlock;
using llvm::Function;
using llvm::Instruction;
using llvm::SmallVector;
using llvm::Use;
using llvm::User;

using llvm::dyn_cast;
using llvm::isa;

template<typename T>
using RPOT = llvm::ReversePostOrderTraversal<T>;

using TypeVector = llvm::SmallVector<model::UpcastableType, 8>;
using ModelTypesMap = std::map<const llvm::Value *,
                               const model::UpcastableType>;

static Logger<> Log{ "init-model-types" };

/// Map each llvm::Argument of the given llvm::Function to its type in the model
static void addArgumentsTypes(const llvm::Function &LLVMFunc,
                              const abi::FunctionType::Layout &Layout,
                              const model::Binary &Model,
                              ModelTypesMap &TypeMap,
                              bool PointersOnly) {

  using AK = abi::FunctionType::ArgumentKind::Values;
  constexpr auto SPTAR = AK::ShadowPointerToAggregateReturnValue;
  auto NonShadowArgs = std::ranges::subrange(Layout.Arguments.begin(),
                                             Layout.Arguments.end());
  if (!Layout.Arguments.empty() && Layout.Arguments[0].Kind == SPTAR) {
    revng_assert(LLVMFunc.arg_size() + 1 == Layout.Arguments.size());
    NonShadowArgs = std::ranges::subrange(std::next(Layout.Arguments.begin()),
                                          Layout.Arguments.end());
  } else {
    revng_assert(LLVMFunc.arg_size() == Layout.Arguments.size());
  }

  for (const auto &[ArgModelType, LLVMArg] :
       llvm::zip_first(NonShadowArgs, LLVMFunc.args())) {
    if (not PointersOnly or ArgModelType.Type->isPointer())
      TypeMap.insert({ &LLVMArg, ArgModelType.Type.copy() });
  }
}

/// Create a type for unvisited operands, i.e. constants, globals and
/// constexprs.
///
/// \return true if a new token has been generated for the operand.
static RecursiveCoroutine<bool> addOperandType(const llvm::Value *Operand,
                                               const model::Binary &Model,
                                               ModelTypesMap &TypeMap,
                                               bool PointersOnly) {
  // For ConstExprs, check their OpCode
  if (auto *Expr = dyn_cast<llvm::ConstantExpr>(Operand)) {
    // A constant expression might have its own uninitialized constant operands
    for (const llvm::Value *Op : Expr->operand_values())
      rc_recur addOperandType(Op, Model, TypeMap, PointersOnly);

    unsigned Opcode = Expr->getOpcode();
    if (Opcode == Instruction::IntToPtr or Opcode == Instruction::PtrToInt) {
      auto It = TypeMap.find(Expr->getOperand(0));
      if (It != TypeMap.end()) {
        const model::UpcastableType &OperandType = It->second;

        if (OperandType->isPointer()) {
          // If the operand is already a pointer, just forward it
          TypeMap.insert({ Operand, OperandType.copy() });
          rc_return true;
        } else if (not PointersOnly) {
          auto PS = model::Architecture::getPointerSize(Model.Architecture());
          TypeMap.insert({ Operand, model::PrimitiveType::makeGeneric(PS) });
          rc_return true;
        }
      }
    }
  } else if (llvm::isa<llvm::ConstantInt>(Operand)
             or llvm::isa<llvm::GlobalVariable>(Operand)) {

    model::UpcastableType Type = modelType(Operand, Model);
    if (not PointersOnly or Type->isPointer()) {
      TypeMap.insert({ Operand, std::move(Type) });
      rc_return true;
    }
  } else if (llvm::isa<llvm::PoisonValue>(Operand)
             or llvm::isa<llvm::UndefValue>(Operand)) {
    // Skip if it's not a pointer and we are only interested in pointers
    if (PointersOnly)
      rc_return false;

    // `poison` and `undef` can either be integers or pointers
    llvm::Type *OperandType = Operand->getType();
    revng_assert(OperandType->isIntOrPtrTy());
    auto ByteSize = model::Architecture::getPointerSize(Model.Architecture());

    model::UpcastableType Result;
    if (auto *IntType = dyn_cast<llvm::IntegerType>(OperandType))
      Result = llvmIntToModelType(IntType, Model);
    else
      Result = model::PrimitiveType::makeGeneric(ByteSize);
    revng_assert(llvm::isa<model::PrimitiveType>(Result.get()));

    TypeMap.insert({ Operand, std::move(Result) });
    rc_return true;

  } else if (auto *NullPtr = dyn_cast<llvm::ConstantPointerNull>(Operand)) {
    if (PointersOnly)
      rc_return false;

    auto PtrSize = model::Architecture::getPointerSize(Model.Architecture());
    TypeMap.insert({ Operand, model::PrimitiveType::makeGeneric(PtrSize) });
    rc_return true;

  } else if (auto *ReferencedFunction = dyn_cast<llvm::Function>(Operand)) {
    if (FunctionTags::Isolated.isTagOf(ReferencedFunction)) {
      // Given a function, obtain a function pointer
      // TODO: introduce helpers, this is terrible
      using namespace model;
      auto EntryAddress = getMetaAddressOfIsolatedFunction(*ReferencedFunction);
      const model::Function &Function = Model.Functions().at(EntryAddress);
      const auto &Prototype = Model.prototypeOrDefault(Function.prototype());
      const auto &Key = Prototype->getPrototype()->key();
      auto PrototypeReference = Model.getDefinitionReference(Key);
      auto PrototypeType = DefinedType::make(PrototypeReference);
      auto PointerSize = Architecture::getPointerSize(Model.Architecture());
      auto Pointer = PointerType::make(std::move(PrototypeType), PointerSize);
      TypeMap.insert({ Operand, Pointer });

      rc_return true;
    }
  }

  rc_return false;
}

/// Reconstruct the return type(s) of a Call instruction from its
/// prototype, if it's an isolated function. For non-isolated functions,
/// special rules apply to recover the returned type.
static TypeVector getReturnTypes(const llvm::CallInst *Call,
                                 const model::Function *ParentFunc,
                                 const model::Binary &Model,
                                 const ModelTypesMap &TypeMap) {
  if (Call->getType()->isVoidTy())
    return {};

  // Check if we already have strong model information for this call
  TypeVector ReturnTypes = getStrongModelInfo(Call, Model);
  if (not ReturnTypes.empty())
    return ReturnTypes;

  auto *CalledFunc = getCalledFunction(Call);
  revng_assert(CalledFunc);

  if (FunctionTags::Parentheses.isTagOf(CalledFunc)
      or FunctionTags::Copy.isTagOf(CalledFunc)
      or FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {

    const llvm::Value *Arg = Call->getArgOperand(0);

    if (auto *ConstInt = dyn_cast<llvm::ConstantInt>(Arg);
        ConstInt and FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
      unsigned BitWidth = ConstInt->getType()->getIntegerBitWidth();
      unsigned ByteSize = std::max(1U, BitWidth / 8U);
      return { model::PrimitiveType::makeSigned(ByteSize) };
    } else {
      // Forward the type
      auto It = TypeMap.find(Arg);
      if (It != TypeMap.end())
        return { It->second };
    }

  } else if (FunctionTags::QEMU.isTagOf(CalledFunc)
             or FunctionTags::Helper.isTagOf(CalledFunc)
             or FunctionTags::Exceptional.isTagOf(CalledFunc)
             or CalledFunc->isIntrinsic()
             or FunctionTags::OpaqueCSVValue.isTagOf(CalledFunc)) {

    revng_assert(not CalledFunc->isTargetIntrinsic());

    llvm::Type *ReturnedType = Call->getType();

    if (ReturnedType->isSingleValueType()) {
      return { llvmIntToModelType(ReturnedType, Model) };

    } else if (ReturnedType->isAggregateType()) {
      // For intrinsics and helpers returning aggregate types, we simply
      // return a list of all the subtypes, after transforming each in the
      // corresponding primitive type
      for (llvm::Type *Subtype : ReturnedType->subtypes())
        ReturnTypes.push_back(llvmIntToModelType(Subtype, Model));

      return ReturnTypes;

    } else {
      revng_abort("Unknown value returned by non-isolated function");
    }

  } else if (FunctionTags::StringLiteral.isTagOf(CalledFunc)) {
    return { model::PointerType::make(model::PrimitiveType::makeUnsigned(1),
                                      Model.Architecture()) };

  } else if (FunctionTags::LiteralPrintDecorator.isTagOf(CalledFunc)) {
    const llvm::Value *Arg = Call->getArgOperand(0);
    return { llvmIntToModelType(Arg->getType(), Model) };

  } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
    return { llvmIntToModelType(Call->getType(), Model) };

  } else if (FunctionTags::BooleanNot.isTagOf(CalledFunc)) {
    return { model::PrimitiveType::makeGeneric(1) };

  } else {
    revng_abort("Unknown non-isolated function");
  }

  return {};
}

/// Given a call instruction, to either an isolated or a non-isolated
/// function, assign to it its return type. If the call returns more than
/// one type, infect the uses of the returned value with those types.
static void handleCallInstruction(const llvm::CallInst *Call,
                                  const model::Function *ParentFunc,
                                  const model::Binary &Model,
                                  ModelTypesMap &TypeMap,
                                  bool PointersOnly) {

  TypeVector ReturnedTypes = getReturnTypes(Call, ParentFunc, Model, TypeMap);
  if (ReturnedTypes.empty())
    return;

  llvm::Type *CallType = Call->getType();
  if (ReturnedTypes.size() == 1) {
    // If the function returns just one value, associate the computed
    // type to the Call Instruction
    revng_assert(not CallType->isStructTy());

    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly or ReturnedTypes[0]->isPointer())
      TypeMap.insert({ Call, ReturnedTypes[0] });

  } else if (not CallType->isAggregateType()) {
    // If we reach this point, we have many types in ReturnedTypes, but the
    // Call on LLVM IR returns an integer.
    revng_assert(CallType->isIntegerTy());
    // In this case we cannot attach a rich type to the integer on LLVM IR, we
    // just have to fall back to a generic primitive
    if (not PointersOnly) {
      auto BitWidth = CallType->getIntegerBitWidth();
      revng_assert(BitWidth > 0 and not(BitWidth % 8));
      TypeMap.insert({ Call, model::PrimitiveType::makeGeneric(BitWidth / 8) });
    }

  } else {
    // If we reach this point, we have many types in ReturnedTypes, and
    // the Call also returns a struct on LLVM IR

    // Functions that return aggregate types have more than one return type.
    // In this case, we cannot assign all the returned types to the returned
    // llvm::Value. Hence, we collect the returned types in a vector and
    // assign them to the values extracted from the returned struct.
    const auto ExtractedValues = getExtractedValuesFromInstruction(Call);
    revng_assert(ReturnedTypes.size() == ExtractedValues.size());

    for (auto &&[Type, ExtractedSet] : zip(ReturnedTypes, ExtractedValues)) {
      revng_assert(Type->isScalar());

      // Each extractedSet contains the set of instructions that extract the
      // same value from the struct
      for (const llvm::CallInst *ExtractValInst : ExtractedSet)
        // Skip if it's not a pointer and we are only interested in pointers
        if (not PointersOnly or Type->isPointer())
          TypeMap.insert({ ExtractValInst, Type.copy() });
    }
  }
}

static model::PrimitiveKind::Values
getCommonPrimitiveKind(model::PrimitiveKind::Values A,
                       model::PrimitiveKind::Values B) {
  if (A == B)
    return A;

  if (A == model::PrimitiveKind::Generic or B == model::PrimitiveKind::Generic)
    return model::PrimitiveKind::Generic;

  // Here, neither A nor B are Generic

  // Given that A != B, and they're not generic, if either of them is Float,
  // we directly go to Generic.
  if (A == model::PrimitiveKind::Float or B == model::PrimitiveKind::Float)
    return model::PrimitiveKind::Generic;

  // Here neither A nor B is Generic nor Float

  // If either is PointerOrNumber, we go to PointerOrNumber.
  if (A == model::PrimitiveKind::PointerOrNumber
      or B == model::PrimitiveKind::PointerOrNumber)
    return model::PrimitiveKind::PointerOrNumber;

  // Here neither A nor B is Generic, Float, nor PointerOrNumber
  // Here A and B can only be Number, Signed or Unsigned.
  // Given that they are different, we always go to Number.
  return model::PrimitiveKind::Number;
}

static model::UpcastableType getCommonScalarType(const model::Type &A,
                                                 const model::Type &B) {
  revng_assert(A.isScalar());
  revng_assert(B.isScalar());

  if (A == B)
    return A;

  revng_assert(A.isPrimitive() or A.isPointer() or A.isEnum());
  revng_assert(B.isPrimitive() or B.isPointer() or B.isEnum());

  revng_assert(A.size() == B.size());
  uint64_t Size = A.size().value();

  const model::PrimitiveType *PrimitiveA = A.getPrimitive();
  const model::PrimitiveType *PrimitiveB = B.getPrimitive();

  if (PrimitiveA and PrimitiveB) {
    auto CommonKind = getCommonPrimitiveKind(PrimitiveA->PrimitiveKind(),
                                             PrimitiveB->PrimitiveKind());
    return model::PrimitiveType::make(CommonKind, Size);

  } else if (PrimitiveA or PrimitiveB) {
    const auto [Primitive, Other] = PrimitiveA ? std::pair{ PrimitiveA, &B } :
                                                 std::pair{ PrimitiveB, &A };

    if (Other->isPointer()) {
      if (Primitive->PrimitiveKind() == model::PrimitiveKind::Generic)
        return *Other;

      else if (Primitive->PrimitiveKind() == model::PrimitiveKind::Float)
        return model::PrimitiveType::makeGeneric(Size);

      else
        return model::PrimitiveType::makePointerOrNumber(Size);

    } else if (const model::EnumDefinition *Enum = Other->getEnum()) {
      auto UnderlyingKind = Enum->underlyingType().PrimitiveKind();
      auto CommonKind = getCommonPrimitiveKind(UnderlyingKind,
                                               Primitive->PrimitiveKind());
      return model::PrimitiveType::make(CommonKind, Size);

    } else {
      revng_abort();
    }

  } else {
    // Here neither A nor B are primitive. They are either enums or pointers.

    const model::EnumDefinition *EnumA = A.getEnum();
    const model::EnumDefinition *EnumB = B.getEnum();
    if (EnumA and EnumB) {
      // Both are enums: make the common integer among the underlying types.
      auto KindA = EnumA->underlyingType().PrimitiveKind();
      auto KindB = EnumB->underlyingType().PrimitiveKind();
      auto CommonKind = getCommonPrimitiveKind(KindA, KindB);
      return model::PrimitiveType::make(CommonKind, Size);

    } else if (A.isPointer() and B.isPointer()) {
      // Make a `void *`
      return model::PointerType::make(model::PrimitiveType::makeVoid(), Size);

    } else {
      // One is a pointer and the other is an enum: we can't find a common
      // type.
      return model::UpcastableType::empty();
    }
  }
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

static RecursiveCoroutine<std::optional<model::UpcastableType>>
initModelTypesImpl(const llvm::Instruction &I,
                   const llvm::Function &F,
                   const model::Function *ModelF,
                   const model::Binary &Model,
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
          revng_assert(Prototype != nullptr);
          auto Ptr = model::PointerType::make(Model.makeType(Prototype->key()),
                                              Model.Architecture());
          TypeMap.insert({ CalledFunction, std::move(Ptr) });
          continue;
        }
      }
      addOperandType(Op, Model, TypeMap, PointersOnly);
    }
  }

  // Insert void types for consistency
  if (InstType->isVoidTy())
    rc_return model::PrimitiveType::makeVoid();

  // Function calls in the IR might correspond to real function calls in
  // the binary or to special intrinsics used by the backend, so they need
  // to be handled separately
  if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
    handleCallInstruction(Call, ModelF, Model, TypeMap, PointersOnly);
    auto CallTypeIt = TypeMap.find(Call);
    if (CallTypeIt != TypeMap.end())
      rc_return CallTypeIt->second.copy();
    else
      rc_return std::nullopt;
  }

  // Only Call instructions can return aggregates
  revng_assert(not InstType->isAggregateType());

  // All ExtractValues should have been converted to OpaqueExtractValue
  revng_assert(not llvm::isa<llvm::ExtractValueInst>(&I));

  switch (I.getOpcode()) {

  case Instruction::Load: {
    auto *Load = dyn_cast<llvm::LoadInst>(&I);

    auto It = TypeMap.find(Load->getPointerOperand());
    if (It == TypeMap.end())
      rc_return std::nullopt;

    const auto &PtrOperandType = *It->second;

    // If the pointer operand is a pointer in the model, we can exploit
    // this information to assign a model type to the loaded value. Note
    // that this makes sense only if the pointee is itself a pointer or a
    // scalar value: if we find a load of N bits from a struct pointer, we
    // don't know if we are loading the entire struct or only some of its
    // fields.
    // TODO: inspect the model to understand if we are loading the first
    // field.
    if (const model::PointerType *Pointer = PtrOperandType.getPointer())
      if (areMemOpCompatible(*Pointer->PointeeType(), *Load->getType(), Model))
        rc_return Pointer->PointeeType();

  } break;

  case Instruction::Alloca: {
    // TODO: eventually AllocaInst will be replaced by calls to
    // revng_local_variable with a type annotation
    llvm::Type *BaseType = llvm::cast<llvm::AllocaInst>(&I)->getAllocatedType();
    revng_assert(BaseType->isSingleValueType());
    rc_return model::PointerType::make(llvmIntToModelType(BaseType, Model),
                                       Model.Architecture());
  }

  case Instruction::Select: {
    auto *Select = dyn_cast<llvm::SelectInst>(&I);
    const auto &Op1Entry = TypeMap.find(Select->getOperand(1));
    const auto &Op2Entry = TypeMap.find(Select->getOperand(2));

    // If the two selected values have the same type, assign that type to
    // the result
    if (Op1Entry != TypeMap.end() and Op2Entry != TypeMap.end()
        and Op1Entry->second == Op2Entry->second)
      rc_return Op1Entry->second;

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
        rc_return It->second;
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
        rc_return It->second;
    }

  } break;

  case Instruction::BitCast:
  case Instruction::Freeze:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt: {
    // Forward the type if there is one
    auto It = TypeMap.find(I.getOperand(0));
    if (It != TypeMap.end()) {
      const model::UpcastableType &OperandType = It->second;
      if (OperandType->isPointer()) {
        rc_return OperandType;
      } else if (not PointersOnly) {
        auto PSize = model::Architecture::getPointerSize(Model.Architecture());
        rc_return model::PrimitiveType::makeGeneric(PSize);
      }
    }
  } break;

  case Instruction::PHI: {
    auto *PHI = llvm::cast<llvm::PHINode>(&I);
    if (bool New = VisitedPHIs.insert(PHI).second) {
      std::optional<model::UpcastableType> Result = std::nullopt;

      llvm::SmallPtrSet<const llvm::Value *, 8>
        NonPHIIncomings = getTransitivePHIIncomings(PHI);

      for (const llvm::Value *Incoming : NonPHIIncomings) {
        std::optional<model::UpcastableType> IncomingType = std::nullopt;
        auto IncomingTypeIt = TypeMap.find(Incoming);
        if (IncomingTypeIt != TypeMap.end()) {
          IncomingType = IncomingTypeIt->second;
        } else if (auto *IncomingInst = dyn_cast<llvm::Instruction>(Incoming)) {
          IncomingType = rc_recur initModelTypesImpl(*IncomingInst,
                                                     F,
                                                     ModelF,
                                                     Model,
                                                     PointersOnly,
                                                     TypeMap,
                                                     VisitedPHIs);
        }

        if (not IncomingType.has_value())
          continue;

        if (not Result.has_value())
          Result = std::move(IncomingType);
        else if (auto C = getCommonScalarType(**Result, **IncomingType))
          Result = std::move(C);
        else
          Result = llvmIntToModelType(PHI->getType(), Model);
      }

      rc_return Result;
    }
  } break;

  default:
    break;
  }

  // We didn't manage to find a suitable type: fall back to the LLVM one.
  rc_return std::nullopt;
}

static RecursiveCoroutine<ModelTypesMap>
initModelTypesImpl(const llvm::Function &F,
                   const model::Function *ModelF,
                   const model::Binary &Model,
                   bool PointersOnly,
                   llvm::SmallPtrSet<const llvm::PHINode *, 8>
                     VisitedPHIs = {}) {

  ModelTypesMap TypeMap;

  const auto *Prototype = Model.prototypeOrDefault(ModelF->prototype());
  auto Layout = abi::FunctionType::Layout::make(*Prototype);
  addArgumentsTypes(F, Layout, Model, TypeMap, PointersOnly);

  for (const BasicBlock *BB : RPOT<const llvm::Function *>(&F)) {
    for (const Instruction &I : *BB) {
      std::optional<model::UpcastableType> Result = rc_recur
        initModelTypesImpl(I,
                           F,
                           ModelF,
                           Model,
                           PointersOnly,
                           TypeMap,
                           VisitedPHIs);
      if (PointersOnly) {
        // Skip if it's not a pointer and we are only interested in pointers
        if (Result.has_value() and !Result->isEmpty()
            and (*Result)->isPointer())
          TypeMap.insert({ &I, std::move(*Result) });

      } else if (Result.has_value()) {
        TypeMap.insert({ &I, std::move(*Result) });

      } else if (I.getType()->isIntOrPtrTy()) {
        // As a fallback, use the LLVM type
        TypeMap.insert({ &I, llvmIntToModelType(I.getType(), Model) });

      } else if (auto *Call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        // TODO: is there more we can check here?

      } else {
        revng_abort("Couldn't process a type.");
      }
    }
  }

  rc_return TypeMap;
}

static bool isUpgradable(const model::UpcastableType &UT) {
  // The only upgradable type is a pointer to void (where
  // neither the pointer nor the pointee doesn't have typedefs).
  const auto *Pointer = dyn_cast<model::PointerType>(&*UT);
  if (not Pointer)
    return false;

  const model::Type &Pointee = Pointer->getPointee();
  return isa<model::PrimitiveType>(Pointee) and Pointee.isVoidPrimitive();
}

static bool isValidScalarUpgrade(const model::UpcastableType &From,
                                 const model::UpcastableType &To) {
  revng_assert(isUpgradable(From));

  if (From == To)
    return true;

  if (From->size() != To->size())
    return false;

  if (From->isNumberPrimitive()) {
    return To->isSignedPrimitive() or To->isUnsignedPrimitive();
  }
  if (From->isPointerOrNumberPrimitive()) {
    return To->isPointer() or To->isNumberPrimitive() or To->isSignedPrimitive()
           or To->isUnsignedPrimitive();
  }
  if (From->isGenericPrimitive()) {
    return not To->isVoidPrimitive();
  }

  // Here From is a non-typedefed pointer to non-typedefed void.
  // Any other pointer is a valid upgrade.
  return To->isPointer();
}

// Returns the most accurate common scalar type among T1 and T2.
// If either T1 or T2 are typedefs, enums, or are *not* scalars, it returns
// nullopt. If either T1 or T2 is empty, it returns the other one. NOTE: this is
// different from getCommonScalarType because it allows for promotion, meaning
// that the common most accurate scalar type among e.g. generic and number is
// number.
static std::optional<model::UpcastableType>
getCommonScalarTypeForPromotion(const model::UpcastableType &T1,
                                const model::UpcastableType &T2) {

  bool T1Empty = T1.isEmpty();
  if (not T1Empty) {
    if (T1->isTypedef())
      return std::nullopt;
    if (T1->isEnum())
      return std::nullopt;
    if (not T1->isScalar())
      return std::nullopt;
  }

  bool T2Empty = T2.isEmpty();
  if (T2Empty) {
    if (T2->isTypedef())
      return std::nullopt;
    if (T2->isEnum())
      return std::nullopt;
    if (not T2->isScalar())
      return std::nullopt;
  }

  if (T1Empty)
    return T2;
  if (T2Empty)
    return T1;

  if (T1->size() != T2->size())
    return std::nullopt;

  auto T1Size = T1->size();
  auto T2Size = T2->size();
  revng_assert(T1Size == T2Size);
  size_t Size = T1Size.value();

  //
  // Handle generic first
  //

  bool T1G = T1->isGenericPrimitive();
  bool T2G = T2->isGenericPrimitive();
  // If both are generic, either one is good.
  if (T1G and T2G)
    return T1;

  // If only one is generic, return the other one, because everything is more
  // accurate than generic, and generic is always compatible with everything
  // else.
  if (T1G or T2G)
    return T1G ? T2 : T1;

  //
  // Then look for floats
  //

  unsigned NumFloats = T1->isFloatPrimitive() + T2->isFloatPrimitive();
  // If both are floats, either is good.
  if (NumFloats == 2)
    return T1;
  // If only one is float, given that we've ruled out generics first, the most
  // accurate type that is compatible with both float and something non-float
  // and non-generic is just generic
  if (NumFloats == 1)
    return model::PrimitiveType::makeGeneric(Size);

  //
  // Then deal with PointerOrNumber
  //

  bool T1NP = T1->isPointerOrNumberPrimitive();
  bool T2NP = T2->isPointerOrNumberPrimitive();
  // If both are PointerOrNumber, either is good.
  if (T1NP and T2NP)
    return T1;
  // If only one is PointerOrNumber, given that we've ruled out generics and
  // floats first, the other one must be at the same time more accurate than
  // PointerOrNumber and compatible with it.
  if (T1NP or T2NP)
    return T1NP ? T2 : T1;

  //
  // Deal with pointers
  //

  unsigned NumPointers = T1->isPointer() + T2->isPointer();
  // If only one is a pointer, given that we've ruled out generics, floats, and
  // PointerOrNumber first, the other one is either Number, Signed or Unsigned.
  // Then, the most accurate that is compatible with both pointers and an
  // integer non-pointer type is a PointerOrNumber
  if (NumPointers == 1)
    return model::PrimitiveType::makePointerOrNumber(Size);
  // If both are pointers we have to look at pointees.
  if (NumPointers == 2) {
    const model::Type &Pointee1 = T1->getPointee();
    const model::Type &Pointee2 = T2->getPointee();
    // If the pointees are the same, either is fine.
    if (Pointee1 == Pointee2)
      return T1;

    bool P1Void = isa<model::PrimitiveType>(Pointee1)
                  and Pointee1.isVoidPrimitive();
    bool P2Void = isa<model::PrimitiveType>(Pointee2)
                  and Pointee2.isVoidPrimitive();
    // If either of the pointees is exactly void * return the other.
    if (P1Void)
      return T2;
    if (P2Void)
      return T1;

    // In all the other cases we have two pointer types, pointing to 2 different
    // types, and none of the pointees is a naked void primitive. In all these
    // cases we return void *.
    return model::PointerType::make(model::PrimitiveType::makeVoid(), Size);
  }

  //
  // Deal with Numbers
  //

  bool T1N = T1->isNumberPrimitive();
  bool T2N = T2->isNumberPrimitive();
  // If both are Number, either is good.
  if (T1N and T2N)
    return T1;
  // If only one is Number, given that we've ruled out generic, floats,
  // pointers, and PointerOrNumber, then the other one must be at the same time
  // more accurate than Number and compatible with it.
  if (T1N or T2N)
    return T1N ? T2 : T1;

  //
  // Deal with Signed
  //

  unsigned NumSigned = T1->isSignedPrimitive() + T2->isSignedPrimitive();

  // If both are signed, either is fine
  if (NumSigned == 2)
    return T1;
  // If only one is Signed, given that we're only left with Signed and Unsigned,
  // the most accurate that is common to both can only be Number
  if (NumSigned == 1)
    return model::PrimitiveType::makeNumber(Size);

  //
  // In principle we're only left with Unsigned here, but make some extra checks
  // to ensure this asserts if in the future we add more kinds of primitives.
  //

  unsigned NumUnsigned = T1->isUnsignedPrimitive() + T2->isUnsignedPrimitive();
  revng_assert(NumUnsigned == 2);
  if (NumUnsigned == 2)
    return T1;

  return std::nullopt;
}

static RecursiveCoroutine<SmallVector<const Use *>>
getUsesThroughCasts(const llvm::Use &U, llvm::ModuleSlotTracker &MST) {
  revng_log(Log, "getUsesThroughCasts");
  LoggerIndent Indent{ Log };

  SmallVector<const Use *> Result;
  User *TheUser = U.getUser();
  if (llvm::isa<llvm::BitCastInst>(TheUser)
      or llvm::isa<llvm::FreezeInst>(TheUser)
      or llvm::isa<llvm::IntToPtrInst>(TheUser)
      or llvm::isa<llvm::PtrToIntInst>(TheUser)) {
    revng_log(Log, "User is a transparent cast, consider its uses");
    LoggerIndent UseIndent{ Log };
    for (const llvm::Use &NestedUse : TheUser->uses()) {
      revng_log(Log,
                "considering use of: "
                  << dumpToString(*NestedUse.get(), MST)
                  << " in : " << dumpToString(*NestedUse.getUser(), MST));
      Result.append(rc_recur getUsesThroughCasts(NestedUse, MST));
    }
  } else {
    Result.push_back(&U);
  }
  rc_return Result;
}

ModelTypesMap initModelTypes(const llvm::Function &F,
                             const model::Function *ModelF,
                             const model::Binary &Model,
                             bool PointersOnly) {
  revng_log(Log, "========= START initModelTypes on " << F.getName());

  ModelTypesMap Result = initModelTypesImpl(F, ModelF, Model, PointersOnly);

  revng_log(Log, "========= END initModelTypes on " << F.getName());
  return Result;
}

ModelTypesMap initModelTypesConsideringUses(const llvm::Function &F,
                                            const model::Function *ModelF,
                                            const model::Binary &Model,
                                            bool PointersOnly) {
  revng_log(Log, "==== START initModelTypesConsideringUses on " << F.getName());

  ModelTypesMap Result = initModelTypes(F, ModelF, Model, PointersOnly);

  llvm::ModuleSlotTracker MST(F.getParent(),
                              /* ShouldInitializeAllMetadata = */ false);
  if (Log.isEnabled())
    MST.incorporateFunction(F);

  // Refine the Result map, trying to upgrade types by looking at their uses and
  // see if they provide more accurate types.
  std::vector<ModelTypesMap::value_type> Upgraded;

  for (auto It = Result.begin(); It != Result.end();) {
    const auto &[V, UT] = *It;

    if (not isa<Instruction>(V) or V->getNumUses() == 0
        or not isUpgradable(UT)) {
      ++It;
      continue;
    }

    revng_log(Log, "try to upgrade the type of: " << dumpToString(*V, MST));
    revng_log(Log, "initial type: " << UT->toString());
    LoggerIndent Indent{ Log };

    // If this optional is ever cleared it means that we cannot compute an
    // accurate common type to all users that is valid, so the substitution
    // should not take place.
    std::optional<model::UpcastableType>
      UpgradedType = model::UpcastableType::empty();

    // If UT is upgradable, we try to promote it, looking at the expected type
    // of all uses of V.
    SmallVector<const Use *> UsesToInspect;
    for (const llvm::Use &U : V->uses()) {
      revng_log(Log,
                "considering use of: " << dumpToString(*U.get(), MST)
                                       << " in : "
                                       << dumpToString(*U.getUser(), MST));
      UsesToInspect.append(getUsesThroughCasts(U, MST));
    }

    for (const llvm::Use *U : UsesToInspect) {
      revng_log(Log,
                "considering use of: " << dumpToString(*U->get(), MST)
                                       << " in : "
                                       << dumpToString(*U->getUser(), MST));
      LoggerIndent UseIndent{ Log };

      SmallVector<model::UpcastableType>
        ExpectedTypes = getExpectedModelType(U, Model);
      revng_assert(ExpectedTypes.size() <= 1);

      if (ExpectedTypes.empty())
        continue;

      auto &UseType = ExpectedTypes[0];

      revng_assert(not UseType.empty());

      revng_log(Log, "ExpectedType: " << UseType->toString());

      UpgradedType = getCommonScalarTypeForPromotion(UpgradedType.value(),
                                                     UseType);

      // If at some point we figure out the Upgraded type cannot be computed we
      // bail out.
      if (not UpgradedType.has_value()) {
        revng_log(Log, "Upgraded type cannot be computed. Bail out.");
        break;
      }
    }

    if (not UpgradedType.has_value()) {
      revng_log(Log, "Upgraded type cannot be computed. Next value.");
      ++It;
      continue;
    }

    if (not UpgradedType->isEmpty() // we hit at use with strong expected type
        and *UpgradedType != UT // we actually need to upgrade something
        and isValidScalarUpgrade(UT, *UpgradedType)) { // the upgrade is valid

      revng_log(Log, "Upgraded to: " << (*UpgradedType)->toString());
      Upgraded.emplace_back(V, UpgradedType->copy());
      It = Result.erase(It);
    } else {
      ++It;
    }
  }

  for (auto &Pair : Upgraded)
    Result.insert({ Pair.first, Pair.second.copy() });

  revng_log(Log, "==== END initModelTypesConsideringUses on " << F.getName());
  return Result;
}
