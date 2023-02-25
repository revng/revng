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

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
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
#include "revng-c/ValueManipulationAnalysis/VMAPipeline.h"

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
    revng_assert(Operand->getType()->isIntOrPtrTy());
    auto ConstType = llvmIntToModelType(Operand->getType(), Model);
    // Skip if it's not a pointer and we are only interested in pointers
    if (not PointersOnly or ConstType.isPointer()) {
      TypeMap.insert({ Operand, ConstType });
    }
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
static TypeVector getReturnTypes(FunctionMetadataCache &Cache,
                                 const llvm::CallInst *Call,
                                 const model::Function *ParentFunc,
                                 const Binary &Model,
                                 ModelTypesMap &TypeMap) {
  TypeVector ReturnTypes;

  if (Call->getType()->isVoidTy())
    return {};

  // Check if we already have strong model information for this call
  ReturnTypes = getStrongModelInfo(Cache, Call, Model);

  if (not ReturnTypes.empty())
    return ReturnTypes;

  auto *CalledFunc = Call->getCalledFunction();
  revng_assert(CalledFunc);

  if (FunctionTags::Parentheses.isTagOf(CalledFunc)
      || FunctionTags::Copy.isTagOf(CalledFunc)
      || FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
    const llvm::Value *Arg = Call->getArgOperand(0);

    // Forward the type
    auto It = TypeMap.find(Arg);
    if (It != TypeMap.end())
      ReturnTypes.push_back(It->second);

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
    const llvm::Value *Arg = Call->getArgOperand(0);
    ReturnTypes.push_back(llvmIntToModelType(Arg->getType(), Model));
  } else if (FunctionTags::HexInteger.isTagOf(CalledFunc)
             || FunctionTags::CharInteger.isTagOf(CalledFunc)
             || FunctionTags::BoolInteger.isTagOf(CalledFunc)) {
    const llvm::Value *Arg = Call->getArgOperand(0);
    ReturnTypes.push_back(llvmIntToModelType(Arg->getType(), Model));
  } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
    ReturnTypes.push_back(llvmIntToModelType(Call->getType(), Model));
  } else {
    revng_abort("Unknown non-isolated function");
  }

  return ReturnTypes;
}

/// Given a call instruction, to either an isolated or a non-isolated
/// function, assign to it its return type. If the call returns more than
/// one type, infect the uses of the returned value with those types.
static void handleCallInstruction(FunctionMetadataCache &Cache,
                                  const llvm::CallInst *Call,
                                  const model::Function *ParentFunc,
                                  const Binary &Model,
                                  ModelTypesMap &TypeMap,
                                  bool PointersOnly) {

  TypeVector ReturnedQualTypes = getReturnTypes(Cache,
                                                Call,
                                                ParentFunc,
                                                Model,
                                                TypeMap);

  if (ReturnedQualTypes.size() == 0)
    return;

  llvm::Type *CallType = Call->getType();
  if (ReturnedQualTypes.size() == 1) {
    // If the function returns just one value, associate the computed
    // QualifiedType to the Call Instruction
    revng_assert(CallType->isSingleValueType());

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
      for (const llvm::ExtractValueInst *ExtractValInst : ExtractedSet)
        // Skip if it's not a pointer and we are only interested in pointers
        if (not PointersOnly or QualType.isPointer())
          TypeMap.insert({ ExtractValInst, QualType });
    }
  }
}

ModelTypesMap initModelTypes(FunctionMetadataCache &Cache,
                             const llvm::Function &F,
                             const model::Function *ModelF,
                             const Binary &Model,
                             bool PointersOnly) {
  ModelTypesMap TypeMap;

  const model::Type *Prototype = ModelF->Prototype().getConst();
  revng_assert(Prototype);

  addArgumentsTypes(F, Prototype, Model, TypeMap, PointersOnly);

  for (const BasicBlock *BB : RPOT<const llvm::Function *>(&F)) {
    for (const Instruction &I : *BB) {

      const auto *InstType = I.getType();

      // Visit operands, in case they are constants, globals or constexprs
      for (const llvm::Value *Op : I.operand_values()) {
        // Ignore operands of some custom opcodes
        if (isCallTo(&I, "revng_call_stack_arguments"))
          continue;
        addOperandType(Op, Model, TypeMap, PointersOnly);
      }

      // Insert void types for consistency
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
        handleCallInstruction(Cache,
                              Call,
                              ModelF,
                              Model,
                              TypeMap,
                              PointersOnly);
        continue;
      }

      // Only Call instructions can return aggregates
      revng_assert(not InstType->isAggregateType());

      // All ExtractValues should have been assigned when handling Call
      // instructions that return an aggregate
      if (isa<llvm::ExtractValueInst>(&I)) {
        if (not PointersOnly)
          revng_assert(TypeMap.contains(&I));
        continue;
      }

      std::optional<QualifiedType> Type;

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

      case Instruction::IntToPtr:
      case Instruction::PtrToInt: {
        // If the PointersOnly flag is set, we ignore IntToPtr and PtrToInt
        if (not PointersOnly) {
          const llvm::Value *Operand = I.getOperand(0);

          // Forward the type if there is one
          auto It = TypeMap.find(Operand);
          if (It != TypeMap.end())
            Type = It->second;
        }

      } break;

      default:
        break;
      }

      if (PointersOnly) {
        // Skip if it's not a pointer and we are only interested in pointers
        if (Type and Type->isPointer())
          TypeMap.insert({ &I, *Type });

      } else {
        // As a fallback, use the LLVM type to build the QualifiedType
        if (not Type)
          Type = llvmIntToModelType(InstType, Model);

        TypeMap.insert({ &I, *Type });
      }
    }
  }

  // Run VMA
  VMAPipeline VMA(Model);
  VMA.addInitializer(std::make_unique<LLVMInitializer>());
  VMA.addInitializer(std::make_unique<TypeMapInitializer>(TypeMap));
  VMA.setUpdater(std::make_unique<TypeMapUpdater>(TypeMap, &Model));
  VMA.disableSolver();

  VMA.run(Cache, &F);

  return TypeMap;
}
