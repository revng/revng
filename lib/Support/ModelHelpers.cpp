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
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using llvm::cast;
using llvm::dyn_cast;

using QualKind = model::QualifierKind::Values;
using CABIFT = model::CABIFunctionDefinition;
using RawFT = model::RawFunctionDefinition;

using model::QualifiedType;
using model::Qualifier;
using model::TypedefDefinition;

constexpr const size_t ModelGEPBaseArgIndex = 1;

static RecursiveCoroutine<model::QualifiedType>
peelConstAndTypedefsImpl(const model::QualifiedType &QT) {
  // First look for non-const qualifiers
  const auto &NonConst = std::not_fn(model::Qualifier::isConst);
  auto QIt = llvm::find_if(QT.Qualifiers(), NonConst);
  auto QEnd = QT.Qualifiers().end();

  // If we find a non-const qualifier we're done unwrapping
  if (QIt != QEnd)
    rc_return model::QualifiedType(QT.UnqualifiedType(), { QIt, QEnd });

  // Here we have only const qualifiers

  auto *TD = dyn_cast<TypedefDefinition>(QT.UnqualifiedType().getConst());

  // If it's not a typedef, we're done. Just throw away the remaining const
  // qualifiers.
  if (not TD)
    rc_return model::QualifiedType(QT.UnqualifiedType(), {});

  // If it's a typedef, unwrap it and recur.
  // Also in this case we can ignore
  rc_return rc_recur peelConstAndTypedefsImpl(TD->UnderlyingType());
}

model::QualifiedType peelConstAndTypedefs(const model::QualifiedType &QT) {
  return peelConstAndTypedefsImpl(QT);
}

static RecursiveCoroutine<model::QualifiedType>
getNonConstImpl(const model::QualifiedType &QT) {
  // First look for non-const qualifiers
  const auto &NonConst = std::not_fn(model::Qualifier::isConst);
  auto QIt = llvm::find_if(QT.Qualifiers(), NonConst);
  auto QEnd = QT.Qualifiers().end();

  // If we find a non-const qualifier we're done unwrapping
  if (QIt != QEnd)
    rc_return model::QualifiedType(QT.UnqualifiedType(), { QIt, QEnd });

  // Here we have only const qualifiers

  auto *TD = dyn_cast<TypedefDefinition>(QT.UnqualifiedType().getConst());

  // If it's not a typedef, we're done. Just throw away the remaining const
  // qualifiers. If it's a typedef but it also doesn't wrap a const type, we are
  // also done.
  if (not TD or not TD->UnderlyingType().isConst())
    rc_return model::QualifiedType(QT.UnqualifiedType(), {});

  // It's a typedef wrapping a const-type, in which case we still have to recur.
  rc_return rc_recur getNonConstImpl(TD->UnderlyingType());
}

model::QualifiedType getNonConst(const model::QualifiedType &QT) {
  return getNonConstImpl(QT);
}

const model::QualifiedType modelType(const llvm::Value *V,
                                     const model::Binary &Model) {
  using namespace llvm;

  Type *T = V->getType();

  bool AddPointer = false;

  // Handle pointers
  if (isa<PointerType>(T)) {
    revng_assert(isa<AllocaInst>(V) or isa<GlobalVariable>(V));
    AddPointer = true;
    T = getVariableType(V);
    revng_assert(isa<IntegerType>(T) or isa<ArrayType>(T));
  } else {
    revng_assert(isa<IntegerType>(T));
  }

  model::QualifiedType Result;

  // Actually build the core type
  if (isa<IntegerType>(T)) {
    Result = llvmIntToModelType(T, Model);
  } else if (auto *Array = dyn_cast<ArrayType>(T)) {
    revng_check(AddPointer);
    Result = llvmIntToModelType(Array->getElementType(), Model);
  }

  revng_assert(Result.UnqualifiedType().isValid());

  // If it was a pointer, add the pointer qualifier
  if (AddPointer)
    Result = Result.getPointerTo(Model.Architecture());

  return Result;
}

const model::QualifiedType llvmIntToModelType(const llvm::Type *LLVMType,
                                              const model::Binary &Model) {
  using namespace model::PrimitiveKind;

  const llvm::Type *TypeToConvert = LLVMType;

  model::QualifiedType ModelType;

  // If it's a pointer, return intptr_t for the current architecture
  //
  // Note: this is suboptimal, in order to avoid this, please use modelType
  // passing the Value instead of invoking llvmIntToModelType passing in just
  // the type
  if (isa<llvm::PointerType>(TypeToConvert)) {
    using namespace model;
    auto Generic = PrimitiveKind::Generic;
    auto PointerSize = Architecture::getPointerSize(Model.Architecture());
    ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, PointerSize);
    return ModelType;
  }

  if (auto *IntType = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    // Convert the integer type
    switch (IntType->getIntegerBitWidth()) {
    case 1:
    case 8:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 1);
      break;

    case 16:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 2);
      break;

    case 32:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 4);
      break;

    case 64:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 8);
      break;

    case 80:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 10);
      break;

    case 96:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 12);
      break;

    case 128:
      ModelType.UnqualifiedType() = Model.getPrimitiveType(Generic, 16);
      break;

    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }
  } else {
    revng_abort("Only integer types can be directly converted from LLVM types "
                "to C types.");
  }

  return ModelType;
}

QualifiedType deserializeFromLLVMString(llvm::Value *V,
                                        const model::Binary &Model) {
  // Try to get a string out of the llvm::Value
  llvm::StringRef BaseTypeString = extractFromConstantStringPtr(V);

  // Try to parse the string as a qualified type (aborts on failure)
  QualifiedType ParsedType;
  {
    llvm::yaml::Input YAMLInput(BaseTypeString);
    YAMLInput >> ParsedType;
    std::error_code EC = YAMLInput.error();
    if (EC)
      revng_abort("Could not deserialize the ModelGEP base type");
  }
  ParsedType.UnqualifiedType().setRoot(&Model);
  revng_assert(ParsedType.UnqualifiedType().isValid());

  return ParsedType;
}

llvm::Constant *serializeToLLVMString(const model::QualifiedType &QT,
                                      llvm::Module &M) {
  // Create a string containing a serialization of the model type
  std::string SerializedQT;
  {
    llvm::raw_string_ostream StringStream(SerializedQT);
    llvm::yaml::Output YAMLOutput(StringStream);
    YAMLOutput << const_cast<model::QualifiedType &>(QT);
  }

  // Build a constant global string containing the serialized type
  return getUniqueString(&M, SerializedQT);
}

RecursiveCoroutine<model::QualifiedType>
dropPointer(const model::QualifiedType &QT) {
  revng_assert(QT.isPointer());

  auto QEnd = QT.Qualifiers().end();
  for (auto QIt = QT.Qualifiers().begin(); QIt != QEnd; ++QIt) {

    if (model::Qualifier::isConst(*QIt))
      continue;

    if (model::Qualifier::isPointer(*QIt)) {
      rc_return model::QualifiedType(QT.UnqualifiedType(),
                                     { std::next(QIt), QEnd });
    } else {
      revng_abort("Error: this is not a pointer");
    }

    rc_return QT;
  }

  // Recur if it has no pointer qualifier but it is a Typedef
  if (auto *TD = dyn_cast<model::TypedefDefinition>(QT.UnqualifiedType().get()))
    rc_return rc_recur dropPointer(TD->UnderlyingType());

  revng_abort("Cannot dropPointer, QT does not have pointer qualifiers");

  rc_return{};
}

static RecursiveCoroutine<QualifiedType>
getFieldType(const QualifiedType &Parent, uint64_t Idx) {
  revng_assert(not Parent.isPointer());

  // If it's an array, we want to discard any const qualifier we have before the
  // first array qualifier, and traverse all typedefs.
  // Pointers are treated as arrays, as if they were traversed by operator []
  if (Parent.isArray() or Parent.isPointer()) {
    QualifiedType Peeled = peelConstAndTypedefs(Parent);

    auto Begin = Peeled.Qualifiers().begin();
    auto End = Peeled.Qualifiers().end();
    revng_assert(Begin != End);
    revng_assert(model::Qualifier::isArray(*Begin)
                 or model::Qualifier::isPointer(*Begin));
    // Then we also throw away the first array qualifier to build a
    // QualifiedType that represents the type of field of the array.
    rc_return model::QualifiedType(Peeled.UnqualifiedType(),
                                   { std::next(Begin), End });
  }

  // If we arrived here, there should be only const qualifiers left
  revng_assert(llvm::all_of(Parent.Qualifiers(), Qualifier::isConst));
  auto *UnqualType = Parent.UnqualifiedType().getConst();

  // Traverse the UnqualifiedType
  if (auto *Struct = dyn_cast<model::StructDefinition>(UnqualType)) {
    rc_return Struct->Fields().at(Idx).Type();
  } else if (auto *Union = dyn_cast<model::UnionDefinition>(UnqualType)) {
    rc_return Union->Fields().at(Idx).Type();
  } else if (auto *Typedef = dyn_cast<model::TypedefDefinition>(UnqualType)) {
    rc_return rc_recur getFieldType(Typedef->UnderlyingType(), Idx);
  }

  revng_abort("Type does not contain fields");
}

static QualifiedType getFieldType(const QualifiedType &Parent,
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

static QualifiedType traverseModelGEP(const model::Binary &Model,
                                      const llvm::CallInst *Call) {
  // Deduce the base type from the first argument
  QualifiedType CurType = deserializeFromLLVMString(Call->getArgOperand(0),
                                                    Model);

  // Compute the first index of variadic arguments that represent the traversal
  // starting from the CurType.
  unsigned IndexOfFirstTraversalArgument = ModelGEPBaseArgIndex + 1;
  if (isCallToTagged(Call, FunctionTags::ModelGEP))
    ++IndexOfFirstTraversalArgument;
  else
    revng_assert(isCallToTagged(Call, FunctionTags::ModelGEPRef));
  // Traverse the model
  for (auto &CurArg :
       llvm::drop_begin(Call->args(), IndexOfFirstTraversalArgument))
    CurType = getFieldType(CurType, CurArg);

  return CurType;
}

llvm::SmallVector<QualifiedType>
flattenReturnTypes(const abi::FunctionType::Layout &Layout,
                   const model::Binary &Model) {

  llvm::SmallVector<QualifiedType> ReturnTypes;

  using namespace abi::FunctionType;
  revng_assert(Layout.returnMethod() == ReturnMethod::RegisterSet);

  auto PointerS = model::Architecture::getPointerSize(Model.Architecture());
  for (const Layout::ReturnValue &ReturnValue : Layout.ReturnValues) {
    if (ReturnValue.Type.isScalar()) {
      if (ReturnValue.Registers.size() > 1) {
        model::QualifiedType PointerSizedInt{
          Model.getPrimitiveType(model::PrimitiveKind::Generic, PointerS), {}
        };

        for (const model::Register::Values &Register : ReturnValue.Registers) {
          revng_assert(model::Register::getSize(Register) == PointerS);
          ReturnTypes.push_back(PointerSizedInt);
        }
      } else {
        ReturnTypes.push_back(ReturnValue.Type);
      }
    } else {
      model::QualifiedType Underlying = peelConstAndTypedefs(ReturnValue.Type);
      revng_assert(Underlying.is(model::TypeDefinitionKind::StructDefinition));
      revng_assert(Underlying.Qualifiers().empty());

      auto *ModelReturnType = Underlying.UnqualifiedType().get();
      auto *StructReturnType = cast<model::StructDefinition>(ModelReturnType);
      for (model::QualifiedType FieldType :
           llvm::map_range(StructReturnType->Fields(),
                           [](const model::StructField &F) {
                             return F.Type();
                           })) {
        revng_assert(FieldType.isScalar());
        ReturnTypes.push_back(std::move(FieldType));
      }
    }
  }

  return ReturnTypes;
}

static llvm::SmallVector<QualifiedType>
handleReturnValue(const model::TypeDefinitionPath &Prototype,
                  const model::Binary &Model) {
  const auto Layout = abi::FunctionType::Layout::make(Prototype);

  switch (Layout.returnMethod()) {
  case abi::FunctionType::ReturnMethod::Void:
    return {};
  case abi::FunctionType::ReturnMethod::ModelAggregate:
    return { Layout.returnValueAggregateType() };
  case abi::FunctionType::ReturnMethod::Scalar:
    revng_assert(Layout.ReturnValues.size() == 1);
    revng_assert(Layout.ReturnValues[0].Type.isScalar());
    return { Layout.ReturnValues[0].Type };
    break;
  case abi::FunctionType::ReturnMethod::RegisterSet:
    return flattenReturnTypes(Layout, Model);
  default:
    revng_abort();
  }
}

RecursiveCoroutine<llvm::SmallVector<QualifiedType>>
getStrongModelInfo(const llvm::Instruction *Inst, const model::Binary &Model) {
  llvm::SmallVector<QualifiedType> ReturnTypes;

  auto ParentFunc = [&Model, &Inst]() {
    return llvmToModelFunction(Model, *Inst->getParent()->getParent());
  };

  if (auto *Call = dyn_cast<llvm::CallInst>(Inst)) {

    if (isCallToIsolatedFunction(Call)) {
      auto Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype.isValid() and not Prototype.empty());

      // Isolated functions and dynamic functions have their prototype in the
      // model
      ReturnTypes = handleReturnValue(Prototype, Model);

    } else {
      // Non-isolated functions do not have a Prototype in the model, but we can
      // infer their returned type(s) in other ways
      auto *CalledFunc = Call->getCalledFunction();
      const auto &FuncName = CalledFunc->getName();
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      if (FuncName.startswith("revng_call_stack_arguments")) {
        auto *Arg0Operand = Call->getArgOperand(0);
        QualifiedType
          CallStackArgumentType = deserializeFromLLVMString(Arg0Operand, Model);
        revng_assert(not CallStackArgumentType.isVoid());

        ReturnTypes.push_back(std::move(CallStackArgumentType));
      } else if (FTags.contains(FunctionTags::ModelGEP)
                 or FTags.contains(FunctionTags::ModelGEPRef)) {
        auto GEPpedType = traverseModelGEP(Model, Call);
        ReturnTypes.push_back(GEPpedType);

      } else if (FTags.contains(FunctionTags::AddressOf)) {
        // The first argument is the base type (not the pointer's type)
        auto Base = deserializeFromLLVMString(Call->getArgOperand(0), Model);
        Base = Base.getPointerTo(Model.Architecture());

        ReturnTypes.push_back(Base);

      } else if (FTags.contains(FunctionTags::ModelCast)
                 or FTags.contains(FunctionTags::LocalVariable)) {
        // The first argument is the returned type
        auto Type = deserializeFromLLVMString(Call->getArgOperand(0), Model);
        ReturnTypes.push_back(Type);

      } else if (FTags.contains(FunctionTags::StructInitializer)) {
        // Struct initializers are only used to pack together return values of
        // RawFunctionTypes that return multiple values, therefore they have
        // the same type as the parent function's return type
        revng_assert(Call->getFunction()->getReturnType() == Call->getType());
        ReturnTypes = handleReturnValue(ParentFunc()->prototype(Model), Model);

      } else if (FTags.contains(FunctionTags::SegmentRef)) {
        const auto &[StartAddress,
                     VirtualSize] = extractSegmentKeyFromMetadata(*CalledFunc);
        auto Segment = Model.Segments().at({ StartAddress, VirtualSize });
        if (not Segment.Type().empty())
          ReturnTypes.push_back(model::QualifiedType{ Segment.Type(), {} });

      } else if (FTags.contains(FunctionTags::Parentheses)) {
        const llvm::Value *Op = Call->getArgOperand(0);
        if (auto *OriginalInst = llvm::dyn_cast<llvm::Instruction>(Op))
          ReturnTypes = rc_recur getStrongModelInfo(OriginalInst, Model);

      } else if (FTags.contains(FunctionTags::OpaqueExtractValue)) {
        const llvm::Value *Op0 = Call->getArgOperand(0);
        if (auto *Aggregate = llvm::dyn_cast<llvm::Instruction>(Op0)) {
          llvm::SmallVector<QualifiedType> NestedReturnTypes = rc_recur
            getStrongModelInfo(Aggregate, Model);
          const auto *Op1 = Call->getArgOperand(1);
          const auto *Index = llvm::cast<llvm::ConstantInt>(Op1);
          ReturnTypes.push_back(NestedReturnTypes[Index->getZExtValue()]);
        }

      } else if (FuncName.startswith("revng_stack_frame")) {
        // Retrieve the stack frame type
        auto &StackType = ParentFunc()->StackFrameType();
        revng_assert(StackType.get());

        ReturnTypes.push_back(QualifiedType{ StackType, {} });

      } else {
        revng_assert(not FuncName.startswith("revng_call_stack_arguments"));
      }
    }
  }
  rc_return ReturnTypes;
}

llvm::SmallVector<QualifiedType>
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
      auto Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype.isValid());

      // If we are inspecting the callee return the prototype
      if (Call->isCallee(U))
        return { createPointerTo(Prototype, Model) };

      if (Call->isArgOperand(U)) {
        const auto Layout = abi::FunctionType::Layout::make(Prototype);
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
      auto Primitive = Model.getPrimitiveType(model::PrimitiveKind::Signed, 8u);
      auto Type = QualifiedType(Primitive,
                                { model::Qualifier::createPointer(8u),
                                  model::Qualifier::createConst() });
      return { Type };
    } else {
      // Non-isolated functions do not have a Prototype in the model, but they
      // can carry type information on their operands
      revng_assert(not Call->isIndirectCall());
      unsigned int ArgOperandIdx = Call->getArgOperandNo(U);

      auto *CalledFunc = Call->getCalledFunction();
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      if (FTags.contains(FunctionTags::AddressOf)
          or FTags.contains(FunctionTags::ModelGEP)
          or FTags.contains(FunctionTags::ModelGEPRef)) {
        // We have model type information only for the base value
        if (ArgOperandIdx != ModelGEPBaseArgIndex)
          return {};

        // The type of the base value is contained in the first operand
        auto Base = deserializeFromLLVMString(Call->getArgOperand(0), Model);
        if (FTags.contains(FunctionTags::ModelGEP))
          Base = Base.getPointerTo(Model.Architecture());
        return { std::move(Base) };

      } else if (isCallTo(Call, "revng_call_stack_arguments")) {
        auto *Arg0Operand = Call->getArgOperand(0);
        QualifiedType
          CallStackArgumentType = deserializeFromLLVMString(Arg0Operand, Model);
        revng_assert(not CallStackArgumentType.isVoid());

        return { std::move(CallStackArgumentType) };
      } else if (FTags.contains(FunctionTags::StructInitializer)) {
        // Struct initializers are only used to pack together return values of
        // RawFunctionTypes that return multiple values, therefore they have
        // the same type as the parent function's return type
        revng_assert(Call->getFunction()->getReturnType() == Call->getType());

        llvm::SmallVector<QualifiedType> ReturnTypes;
        ReturnTypes = handleReturnValue(ParentFunc()->prototype(Model), Model);
        return { ReturnTypes[ArgOperandIdx] };
      } else if (FTags.contains(FunctionTags::BinaryNot)) {
        return { llvmIntToModelType(Call->getType(), Model) };
      }
    }
  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(User)) {
    return handleReturnValue(ParentFunc()->prototype(Model), Model);
  } else if (auto *BinaryOp = dyn_cast<llvm::BinaryOperator>(User)) {
    using namespace model::PrimitiveKind;
    auto Opcode = BinaryOp->getOpcode();
    switch (Opcode) {

    case llvm::Instruction::SDiv:
    case llvm::Instruction::SRem: {
      model::QualifiedType Result;
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));
      auto Bytes = BitWidth / 8;
      Result.UnqualifiedType() = Model.getPrimitiveType(Signed, Bytes);
      return { Result };
    } break;

    case llvm::Instruction::UDiv:
    case llvm::Instruction::URem: {
      model::QualifiedType Result;
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));
      auto Bytes = BitWidth / 8;
      Result.UnqualifiedType() = Model.getPrimitiveType(Unsigned, Bytes);
      return { Result };
    } break;

    case llvm::Instruction::AShr:
    case llvm::Instruction::LShr:
    case llvm::Instruction::Shl: {
      model::QualifiedType Result;
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(BitWidth >= 8 and std::has_single_bit(BitWidth));
      auto Bytes = BitWidth / 8;

      if (U->getOperandNo() == 0) {
        switch (Opcode) {
        case llvm::Instruction::AShr:
          Result.UnqualifiedType() = Model.getPrimitiveType(Signed, Bytes);
          break;

        case llvm::Instruction::LShr:
          Result.UnqualifiedType() = Model.getPrimitiveType(Unsigned, Bytes);
          break;

        case llvm::Instruction::Shl:
          Result.UnqualifiedType() = Model.getPrimitiveType(Number, Bytes);
          break;

        default:
          revng_abort();
        }
      }

      if (U->getOperandNo() == 1)
        Result.UnqualifiedType() = Model.getPrimitiveType(Unsigned, Bytes);

      return { Result };
    } break;
    case llvm::Instruction::Sub:
    case llvm::Instruction::Add: {
      model::QualifiedType Result;
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(std::has_single_bit(BitWidth)
                   and (BitWidth == 1 or BitWidth >= 8));
      auto Bytes = (BitWidth == 1) ? 1 : BitWidth / 8;
      // The second operand of sub should be a number.
      if (Opcode == llvm::Instruction::Sub and U->getOperandNo() == 1)
        Result.UnqualifiedType() = Model.getPrimitiveType(Number, Bytes);
      else
        Result.UnqualifiedType() = Model.getPrimitiveType(PointerOrNumber,
                                                          Bytes);
      return { Result };
    } break;
    case llvm::Instruction::Mul:
    case llvm::Instruction::And:
    case llvm::Instruction::Or:
    case llvm::Instruction::Xor: {
      model::QualifiedType Result;
      auto BitWidth = U->get()->getType()->getIntegerBitWidth();
      revng_assert(std::has_single_bit(BitWidth)
                   and (BitWidth == 1 or BitWidth >= 8));
      auto Bytes = (BitWidth == 1) ? 1 : BitWidth / 8;
      Result.UnqualifiedType() = Model.getPrimitiveType(Number, Bytes);
      return { Result };
    } break;
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
      model::QualifiedType Result;
      using model::PrimitiveKind::PointerOrNumber;
      auto PointerSize = model::Architecture::getPointerSize(Model
                                                               .Architecture());
      Result.UnqualifiedType() = Model.getPrimitiveType(PointerOrNumber,
                                                        PointerSize);
      return { Result };
    }

    // If we're not doing eq or neq, we have to make sure that the
    // signedness is compatible, otherwise it would break semantics.
    using model::PrimitiveKind::PointerOrNumber;
    using model::PrimitiveKind::Signed;
    using model::PrimitiveKind::Unsigned;
    auto ICmpKind = ICmp->isEquality() ? PointerOrNumber :
                                         (ICmp->isSigned() ? Signed : Unsigned);

    auto DL = ICmp->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Op0->getType());

    auto TargetType = model::QualifiedType(Model.getPrimitiveType(ICmpKind,
                                                                  ByteSize),
                                           {});
    return { TargetType };

  } else if (auto *Select = dyn_cast<llvm::SelectInst>(User)) {

    auto DL = Select->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Select->getOperand(1)->getType());
    model::QualifiedType Result;
    using model::PrimitiveKind::Generic;
    Result.UnqualifiedType() = Model.getPrimitiveType(Generic, ByteSize);
    return { Result };

  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(User)) {

    auto DL = Switch->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(Switch->getCondition()->getType());
    model::QualifiedType Result;
    using model::PrimitiveKind::Number;
    Result.UnqualifiedType() = Model.getPrimitiveType(Number, ByteSize);
    return { Result };

  } else if (auto *Trunc = dyn_cast<llvm::TruncInst>(User)) {

    llvm::Type *ResultTy = Trunc->getType();
    auto DL = Trunc->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    model::QualifiedType Result;
    using model::PrimitiveKind::Number;
    Result.UnqualifiedType() = Model.getPrimitiveType(Number, ByteSize);
    return { Result };

  } else if (auto *SExt = dyn_cast<llvm::SExtInst>(User)) {

    llvm::Type *ResultTy = SExt->getType();
    auto DL = SExt->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    model::QualifiedType Result;
    using model::PrimitiveKind::Signed;
    Result.UnqualifiedType() = Model.getPrimitiveType(Signed, ByteSize);
    return { Result };

  } else if (auto *ZExt = dyn_cast<llvm::ZExtInst>(User)) {

    llvm::Type *ResultTy = ZExt->getType();
    auto DL = ZExt->getModule()->getDataLayout();
    uint64_t ByteSize = DL.getTypeAllocSize(ResultTy);
    model::QualifiedType Result;
    using model::PrimitiveKind::Unsigned;
    Result.UnqualifiedType() = Model.getPrimitiveType(Unsigned, ByteSize);
    return { Result };
  }

  return {};
}
