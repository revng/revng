//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using llvm::dyn_cast;

constexpr const size_t ModelGEPBaseArgIndex = 1;

model::UpcastableType modelType(const llvm::Value *V,
                                const model::Binary &Model) {
  model::UpcastableType Result;

  using namespace llvm;

  llvm::Type *T = V->getType();

  // Handle pointers
  bool AddPointer = false;
  if (isa<llvm::PointerType>(T)) {
    revng_assert(isa<llvm::AllocaInst>(V) or isa<llvm::GlobalVariable>(V));
    AddPointer = true;
    T = getVariableType(V);
    revng_assert(isa<llvm::IntegerType>(T) or isa<llvm::ArrayType>(T));
  } else {
    revng_assert(isa<llvm::IntegerType>(T));
  }

  // Actually build the core type
  if (isa<llvm::IntegerType>(T)) {
    Result = llvmIntToModelType(T, Model);
  } else if (auto *Array = llvm::dyn_cast<llvm::ArrayType>(T)) {
    revng_check(AddPointer);
    Result = llvmIntToModelType(Array->getElementType(), Model);
  }

  revng_assert(Result->verify());

  // If it is a pointer, make sure to mark is as such
  if (AddPointer)
    return model::PointerType::make(std::move(Result), Model.Architecture());
  else
    return Result;
}

model::UpcastableType llvmIntToModelType(const llvm::Type *TypeToConvert,
                                         const model::Binary &Model) {
  model::UpcastableType Result = model::UpcastableType::empty();
  if (isa<llvm::PointerType>(TypeToConvert)) {
    // If it's a pointer, return intptr_t for the current architecture
    //
    // Note: this is suboptimal, in order to avoid this, please use modelType
    // passing the Value instead of invoking llvmIntToModelType passing in just
    // the type

    auto PtrSize = model::Architecture::getPointerSize(Model.Architecture());
    Result = model::PrimitiveType::makeGeneric(PtrSize);
  }

  if (auto *Int = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    // Convert the integer type
    if (Int->getIntegerBitWidth() == 1) {
      Result = model::PrimitiveType::makeGeneric(1);
    } else {
      revng_assert(Int->getIntegerBitWidth() % 8 == 0);
      Result = model::PrimitiveType::makeGeneric(Int->getIntegerBitWidth() / 8);
    }
  }

  if (Result.isEmpty()) {
    revng_abort("Only integer and pointer types can be directly converted from "
                "LLVM types to C types.");
  }

  revng_assert(Result->verify(true),
               ("Unsupported llvm type: " + toString(Result)).c_str());
  return Result;
}

model::UpcastableType fromLLVMString(llvm::Value *V,
                                     const model::Binary &Model) {
  // Try to get a string out of the llvm::Value
  llvm::StringRef BaseTypeString = extractFromConstantStringPtr(V);
  auto ParsedType = fromString<model::UpcastableType>(BaseTypeString);
  if (not ParsedType) {
    std::string Error = "Could not deserialize the model type from LLVM "
                        "constant string \""
                        + BaseTypeString.str()
                        + "\": " + consumeToString(ParsedType) + ".";
    revng_abort(Error.c_str());
  }

  revng_assert(!ParsedType->isEmpty(),
               "Type in a LLVM constant string was set to "
               "`model::UpcastableType::empty()`. How did it slip through?");

  if (model::DefinedType *Defined = (*ParsedType)->skipToDefinedType()) {
    model::DefinitionReference &Reference = Defined->Definition();

    revng_assert(Reference.isValid() == false);
    Reference.setRoot(&Model);
    revng_assert(Reference.isValid() == true);
    revng_assert(Reference.getConst() != nullptr);
  } else {
    // Primitives have no references, so no need to do anything special.
  }
  revng_assert((*ParsedType)->verify(true));

  return *ParsedType;
}

llvm::Constant *toLLVMString(const model::UpcastableType &Type,
                             llvm::Module &M) {
  return getUniqueString(&M, toString(Type));
}

static const model::Type &getFieldType(const model::Type &Parent,
                                       uint64_t Idx) {
  const model::Type &Unwrapped = *Parent.skipConstAndTypedefs();
  revng_assert(not Unwrapped.isPointer());

  // If it's an array, we can just return its element type.
  if (const model::ArrayType *Array = Unwrapped.getArray())
    return *Array->ElementType();

  // If we get to this point, the type is neither a pointer nor an array,
  // so it must be either a struct or a union.
  revng_assert(llvm::isa<model::DefinedType>(Unwrapped));
  if (auto *Struct = Unwrapped.getStruct())
    return *Struct->Fields().at(Idx).Type();
  else if (auto *Union = Unwrapped.getUnion())
    return *Union->Fields().at(Idx).Type();

  revng_abort("Type does not contain fields");
}

static const model::Type &getFieldType(const model::Type &Parent,
                                       llvm::Value *Idx) {
  revng_assert(not Parent.isPointer());

  uint64_t NumericIdx = 0;

  if (auto *ArgAsInt = dyn_cast<llvm::ConstantInt>(Idx)) {
    // If the value is a constant integer, use that as index
    NumericIdx = ArgAsInt->getValue().getLimitedValue();
  } else {
    // If the index is not an integer, we can only be traversing an array. In
    // that case, since all elements of an array have the same type, we are not
    // interested in the numeric value of the index. So, we leave it at 0.
    revng_assert(Parent.isArray());
  }

  return getFieldType(Parent, NumericIdx);
}

static model::UpcastableType traverseModelGEP(const model::Binary &Model,
                                              const llvm::CallInst *Call) {
  // Deduce the base type from the first argument
  auto Type = fromLLVMString(Call->getArgOperand(0), Model);

  // Compute the first index of variadic arguments that represent the traversal
  // starting from the CurType.
  unsigned IndexOfFirstTraversalArgument = ModelGEPBaseArgIndex + 1;
  if (isCallToTagged(Call, FunctionTags::ModelGEP))
    ++IndexOfFirstTraversalArgument;
  else
    revng_assert(isCallToTagged(Call, FunctionTags::ModelGEPRef));

  // Traverse the model
  const model::Type *Result = Type.get();
  for (auto &CurArg :
       llvm::drop_begin(Call->args(), IndexOfFirstTraversalArgument)) {
    Result = &getFieldType(*Result, CurArg);
  }

  return *Result;
}

llvm::SmallVector<model::UpcastableType>
flattenReturnTypes(const abi::FunctionType::Layout &Layout,
                   const model::Binary &Model) {

  llvm::SmallVector<model::UpcastableType> ReturnTypes;

  using namespace abi::FunctionType;
  revng_assert(Layout.returnMethod() == ReturnMethod::RegisterSet);

  auto PointerS = model::Architecture::getPointerSize(Model.Architecture());
  for (const Layout::ReturnValue &ReturnValue : Layout.ReturnValues) {
    if (ReturnValue.Type->isScalar()) {
      if (ReturnValue.Registers.size() > 1) {
        for (const model::Register::Values &Register : ReturnValue.Registers) {
          revng_assert(model::Register::getSize(Register) == PointerS);
          ReturnTypes.push_back(model::PrimitiveType::makeGeneric(PointerS));
        }
      } else {
        ReturnTypes.push_back(ReturnValue.Type);
      }
    } else {
      auto GetFieldType = std::views::transform([](const auto &F) {
        return F.Type();
      });

      const auto &StructReturnType = ReturnValue.Type->toStruct();
      for (const auto &FieldType : StructReturnType.Fields() | GetFieldType) {
        revng_assert(FieldType->isScalar());
        ReturnTypes.push_back(std::move(FieldType));
      }
    }
  }

  return ReturnTypes;
}

static llvm::SmallVector<model::UpcastableType>
handleReturnValue(const model::TypeDefinition &Prototype,
                  const model::Binary &Model) {
  const auto Layout = abi::FunctionType::Layout::make(Prototype);

  switch (Layout.returnMethod()) {
  case abi::FunctionType::ReturnMethod::Void:
    return {};
  case abi::FunctionType::ReturnMethod::ModelAggregate:
    return { Layout.returnValueAggregateType() };
  case abi::FunctionType::ReturnMethod::Scalar:
    revng_assert(Layout.ReturnValues.size() == 1);
    revng_assert(Layout.ReturnValues[0].Type->isScalar());
    return { Layout.ReturnValues[0].Type };
  case abi::FunctionType::ReturnMethod::RegisterSet:
    return flattenReturnTypes(Layout, Model);
  default:
    revng_abort();
  }
}

RecursiveCoroutine<llvm::SmallVector<model::UpcastableType, 8>>
getStrongModelInfo(const llvm::Instruction *Inst, const model::Binary &Model) {

  if (auto *Call = dyn_cast<llvm::CallInst>(Inst)) {

    if (isCallToIsolatedFunction(Call)) {
      const auto *Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype != nullptr);

      // Isolated functions and dynamic functions have their prototype in the
      // model
      rc_return handleReturnValue(*Prototype, Model);

    } else {
      // Non-isolated functions do not have a Prototype in the model, but we can
      // infer their returned type(s) in other ways
      auto *CalledFunc = getCalledFunction(Call);
      const auto &FuncName = CalledFunc->getName();
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      auto ParentFunc = [&Model, &Inst]() {
        return llvmToModelFunction(Model, *Inst->getParent()->getParent());
      };

      if (FuncName.startswith("revng_call_stack_arguments")) {
        auto *Arg0Operand = Call->getArgOperand(0);
        auto CallStackArgumentType = fromLLVMString(Arg0Operand, Model);
        revng_assert(not CallStackArgumentType->isVoidPrimitive());

        rc_return{ CallStackArgumentType };
      } else if (FTags.contains(FunctionTags::ModelGEP)
                 or FTags.contains(FunctionTags::ModelGEPRef)) {
        rc_return{ traverseModelGEP(Model, Call) };

      } else if (FTags.contains(FunctionTags::AddressOf)) {
        // The first argument is the base type (not the pointer's type)
        auto Base = fromLLVMString(Call->getArgOperand(0), Model);
        rc_return{ model::PointerType::make(std::move(Base),
                                            Model.Architecture()) };

      } else if (FTags.contains(FunctionTags::ModelCast)
                 or FTags.contains(FunctionTags::LocalVariable)) {
        // The first argument is the returned type
        auto Type = fromLLVMString(Call->getArgOperand(0), Model);
        rc_return{ std::move(Type) };

      } else if (FTags.contains(FunctionTags::StructInitializer)) {
        // Struct initializers are only used to pack together return values of
        // RawFunctionTypes that return multiple values, therefore they have
        // the same type as the parent function's return type
        revng_assert(Call->getFunction()->getReturnType() == Call->getType());
        auto &Prototype = *Model.prototypeOrDefault(ParentFunc()->prototype());
        rc_return handleReturnValue(Prototype, Model);

      } else if (FTags.contains(FunctionTags::SegmentRef)) {
        const auto &[StartAddress,
                     VirtualSize] = extractSegmentKeyFromMetadata(*CalledFunc);
        auto Segment = Model.Segments().at({ StartAddress, VirtualSize });
        if (not Segment.Type().isEmpty())
          rc_return{ Segment.Type() };

      } else if (FTags.contains(FunctionTags::Parentheses)) {
        const llvm::Value *Op = Call->getArgOperand(0);
        if (auto *OriginalInst = llvm::dyn_cast<llvm::Instruction>(Op))
          rc_return rc_recur getStrongModelInfo(OriginalInst, Model);

      } else if (FTags.contains(FunctionTags::OpaqueExtractValue)) {
        const llvm::Value *Op0 = Call->getArgOperand(0);
        if (auto *Aggregate = llvm::dyn_cast<llvm::Instruction>(Op0)) {
          llvm::SmallVector NestedRVs = rc_recur getStrongModelInfo(Aggregate,
                                                                    Model);
          const auto *Op1 = Call->getArgOperand(1);
          const auto *Index = llvm::cast<llvm::ConstantInt>(Op1);
          rc_return{ NestedRVs[Index->getZExtValue()] };
        }

      } else if (FuncName.startswith("revng_stack_frame")) {
        // Retrieve the stack frame type
        revng_assert(not ParentFunc()->StackFrameType().isEmpty());
        rc_return{ ParentFunc()->StackFrameType() };

      } else {
        revng_assert(not FuncName.startswith("revng_call_stack_arguments"));
      }
    }
  }

  rc_return{};
}

llvm::SmallVector<model::UpcastableType>
getExpectedModelType(const llvm::Use *U, const model::Binary &Model) {
  llvm::Instruction *User = dyn_cast<llvm::Instruction>(U->getUser());

  if (not User)
    return {};

  auto ParentFunc = [&Model, &User]() {
    return llvmToModelFunction(Model, *User->getParent()->getParent());
  };

  if (auto *Call = dyn_cast<llvm::CallInst>(User)) {
    if (isCallToIsolatedFunction(Call)) {
      // Isolated functions have their prototype in the model
      const auto *Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype != nullptr);

      // If we are inspecting the callee return the prototype
      if (Call->isCallee(U))
        return { model::PointerType::make(Model.makeType(Prototype->key()),
                                          Model.Architecture()) };

      if (Call->isArgOperand(U)) {
        const auto Layout = abi::FunctionType::Layout::make(*Prototype);
        auto ArgNo = Call->getArgOperandNo(U);

        const auto IsNonShadow =
          [](const abi::FunctionType::Layout::Argument &A) {
            using namespace abi::FunctionType::ArgumentKind;
            return A.Kind != ShadowPointerToAggregateReturnValue;
          };
        auto NonShadowArgs = llvm::make_filter_range(Layout.Arguments,
                                                     IsNonShadow);

        for (const auto &ArgType : llvm::enumerate(NonShadowArgs))
          if (ArgType.index() == ArgNo)
            return { ArgType.value().Type };
        revng_abort();
      }
    } else if (isCallToTagged(Call, FunctionTags::StringLiteral)) {
      return {
        model::PointerType::make(model::PrimitiveType::makeConstSigned(8),
                                 Model.Architecture())
      };
    } else {
      // Non-isolated functions do not have a Prototype in the model, but they
      // can carry type information on their operands
      revng_assert(not Call->isIndirectCall());
      unsigned int ArgOperandIdx = Call->getArgOperandNo(U);

      auto *CalledFunc = getCalledFunction(Call);
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      if (FTags.contains(FunctionTags::AddressOf)) {
        // We have model type information only for the base value
        if (ArgOperandIdx != 1)
          return {};

        // The type of the base value is contained in the first operand
        auto Base = fromLLVMString(Call->getArgOperand(0), Model);
        if (FTags.contains(FunctionTags::ModelGEP))
          Base = model::PointerType::make(std::move(Base),
                                          Model.Architecture());
        return { std::move(Base) };

      } else if (FTags.contains(FunctionTags::ModelGEP)
                 or FTags.contains(FunctionTags::ModelGEPRef)) {
        // We have model type information only for the base value
        if (ArgOperandIdx < ModelGEPBaseArgIndex)
          return {};

        if (ArgOperandIdx == ModelGEPBaseArgIndex) {
          // The type of the base value is contained in the first operand
          auto Base = fromLLVMString(Call->getArgOperand(0), Model);
          if (FTags.contains(FunctionTags::ModelGEP))
            Base = model::PointerType::make(std::move(Base),
                                            Model.Architecture());
          return { std::move(Base) };
        } else {
          // For all index operands in ModelGEP, if the operand is not an
          // integer constant it must be an array index, for which the expected
          // type is a signed integer.
          if (not isa<llvm::ConstantInt>(U->get())) {
            unsigned BitSize = U->get()->getType()->getScalarSizeInBits();
            revng_assert(BitSize);
            revng_assert(BitSize == 1 or 0 == (BitSize % 8));
            model::UpcastableType
              Result = model::PrimitiveType::makeNumber(BitSize == 1 ?
                                                          BitSize :
                                                          BitSize / 8);
            if (not Result->verify()) {
              using model::Architecture::getPointerSize;
              size_t PointerSize = getPointerSize(Model.Architecture());
              Result = model::PrimitiveType::makeNumber(PointerSize);
              revng_assert(Result->verify());
            }
            return { std::move(Result) };
          }
          return {};
        }

      } else if (isCallTo(Call, "revng_call_stack_arguments")) {
        auto *Arg0Operand = Call->getArgOperand(0);
        auto CallStackArgumentType = fromLLVMString(Arg0Operand, Model);
        revng_assert(not CallStackArgumentType.isEmpty());

        return { std::move(CallStackArgumentType) };
      } else if (FTags.contains(FunctionTags::StructInitializer)) {
        // Struct initializers are only used to pack together return values of
        // RawFunctionTypes that return multiple values, therefore they have
        // the same type as the parent function's return type
        revng_assert(Call->getFunction()->getReturnType() == Call->getType());

        auto &Prototype = *Model.prototypeOrDefault(ParentFunc()->prototype());
        return { handleReturnValue(Prototype, Model)[ArgOperandIdx] };
      } else if (FTags.contains(FunctionTags::BinaryNot)) {
        return { llvmIntToModelType(Call->getType(), Model) };
      }
    }
  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(User)) {
    auto &Prototype = *Model.prototypeOrDefault(ParentFunc()->prototype());
    return handleReturnValue(Prototype, Model);
  } else if (auto *BinaryOp = dyn_cast<llvm::BinaryOperator>(User)) {
    using namespace model::PrimitiveKind;
    auto Opcode = BinaryOp->getOpcode();
    switch (Opcode) {

    case llvm::Instruction::SDiv:
    case llvm::Instruction::SRem: {
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));
      return { model::PrimitiveType::makeSigned(BitWidth / 8) };
    }

    case llvm::Instruction::UDiv:
    case llvm::Instruction::URem: {
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));
      return { model::PrimitiveType::makeUnsigned(BitWidth / 8) };
    }

    case llvm::Instruction::AShr:
    case llvm::Instruction::LShr:
    case llvm::Instruction::Shl: {
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));

      if (U->getOperandNo() == 0) {
        switch (Opcode) {
        case llvm::Instruction::AShr:
          return { model::PrimitiveType::makeSigned(BitWidth / 8) };

        case llvm::Instruction::LShr:
          return { model::PrimitiveType::makeUnsigned(BitWidth / 8) };

        case llvm::Instruction::Shl:
          return { model::PrimitiveType::makeNumber(BitWidth / 8) };

        default:
          revng_abort();
        }
      }

      if (U->getOperandNo() == 1)
        return { model::PrimitiveType::makeUnsigned(BitWidth / 8) };

    } break;
    case llvm::Instruction::Sub:
    case llvm::Instruction::Add: {
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(std::has_single_bit(BitWidth)
                   and (BitWidth == 1 or BitWidth >= 8));
      auto Bytes = (BitWidth == 1) ? 1 : BitWidth / 8;
      // The second operand of sub should be a number.
      if (Opcode == llvm::Instruction::Sub and U->getOperandNo() == 1)
        return { model::PrimitiveType::makeNumber(Bytes) };
      else
        return { model::PrimitiveType::makePointerOrNumber(Bytes) };
    }
    case llvm::Instruction::Mul:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor: {
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(std::has_single_bit(BitWidth)
                   and (BitWidth == 1 or BitWidth >= 8));
      auto Bytes = (BitWidth == 1) ? 1 : BitWidth / 8;
      return { model::PrimitiveType::makeNumber(Bytes) };
    }
    case llvm::Instruction::FAdd:
    case llvm::Instruction::FSub:
    case llvm::Instruction::FMul:
    case llvm::Instruction::FDiv:
    case llvm::Instruction::FRem: {
      revng_abort("unexpected floating point binary operation");
    }
    default:
      // no strict requirement for others
      ;
    }
  } else if (auto *ICmp = dyn_cast<llvm::ICmpInst>(User)) {
    const llvm::Value *Op0 = ICmp->getOperand(0);
    const llvm::Value *Op1 = ICmp->getOperand(1);

    // If any of the operands is a pointer, we assume that both of operands are
    // pointers.
    if (Op0->getType()->isPointerTy() or Op1->getType()->isPointerTy()) {
      auto PSize = model::Architecture::getPointerSize(Model.Architecture());
      return { model::PrimitiveType::makePointerOrNumber(PSize) };
    }

    // If we're not doing eq or neq, we have to make sure that the
    // signedness is compatible, otherwise it would break semantics.
    auto ICmpKind = ICmp->isEquality() ?
                      model::PrimitiveKind::PointerOrNumber :
                      (ICmp->isSigned() ? model::PrimitiveKind::Signed :
                                          model::PrimitiveKind::Unsigned);

    auto DL = ICmp->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Op0->getType());
    return { model::PrimitiveType::make(ICmpKind, ByteSize) };

  } else if (auto *Select = dyn_cast<llvm::SelectInst>(User)) {

    auto DL = Select->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Select->getOperand(1)->getType());
    return { model::PrimitiveType::makeGeneric(ByteSize) };

  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(User)) {

    auto DL = Switch->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Switch->getCondition()->getType());
    return { model::PrimitiveType::makeNumber(ByteSize) };

  } else if (auto *Trunc = dyn_cast<llvm::TruncInst>(User)) {

    llvm::Type *ResultTy = Trunc->getType();
    auto DL = Trunc->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    return { model::PrimitiveType::makeNumber(ByteSize) };

  } else if (auto *SExt = dyn_cast<llvm::SExtInst>(User)) {

    llvm::Type *ResultTy = SExt->getType();
    auto DL = SExt->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    return { model::PrimitiveType::makeSigned(ByteSize) };

  } else if (auto *ZExt = dyn_cast<llvm::ZExtInst>(User)) {

    llvm::Type *ResultTy = ZExt->getType();
    auto DL = ZExt->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    return { model::PrimitiveType::makeUnsigned(ByteSize) };
  }

  return {};
}
