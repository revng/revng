//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ABI/FunctionType.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"

using llvm::cast;
using llvm::dyn_cast;

using QualKind = model::QualifierKind::Values;
using CABIFT = model::CABIFunctionType;
using RawFT = model::RawFunctionType;

using model::QualifiedType;
using model::Qualifier;
using model::TypedefType;

constexpr const size_t ModelGEPBaseArgIndex = 1;

model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT, model::VerifyHelper &VH) {
  model::QualifiedType Result = QT;

  while (true) {

    const auto &NonConst = std::not_fn(model::Qualifier::isConst);
    auto QIt = llvm::find_if(Result.Qualifiers, NonConst);
    auto QEnd = Result.Qualifiers.end();

    if (QIt != QEnd)
      return model::QualifiedType(QT.UnqualifiedType, { QIt, QEnd });

    // If we reach this point, Result has no pointer nor array qualifiers.
    // Let's look at the Unqualified type and unwrap it if it's a typedef.
    if (auto *TD = dyn_cast<TypedefType>(Result.UnqualifiedType.getConst())) {
      Result = model::QualifiedType(Result.UnqualifiedType, {});
    } else {
      // If Result is not a typedef we can bail out
      break;
    }
  }
  revng_assert(Result.verify(VH));
  return Result;
}

const model::QualifiedType
llvmIntToModelType(const llvm::Type *LLVMType, const model::Binary &Model) {
  using namespace model::PrimitiveTypeKind;

  const llvm::Type *TypeToConvert = LLVMType;
  size_t NPtrQualifiers = 0;

  // If it's a pointer, find the pointed type
  while (auto *PtrType = dyn_cast<llvm::PointerType>(TypeToConvert)) {
    TypeToConvert = PtrType->getElementType();
    ++NPtrQualifiers;
  }

  model::QualifiedType ModelType;

  if (auto *IntType = dyn_cast<llvm::IntegerType>(TypeToConvert)) {
    // Convert the integer type
    switch (IntType->getIntegerBitWidth()) {
    case 1:
    case 8:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, 1);
      break;

    case 16:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, 2);
      break;

    case 32:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, 4);
      break;

    case 64:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, 8);
      break;

    case 128:
      ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, 16);
      break;

    default:
      revng_abort("Found an LLVM integer with a size that is not a power of "
                  "two");
    }
    // Add qualifiers
    for (size_t I = 0; I < NPtrQualifiers; ++I)
      addPointerQualifier(ModelType, Model);

  } else if (NPtrQualifiers > 0) {
    // If it's a pointer to a non-integer type, return an integer type of the
    // length of a pointer
    auto PtrSize = getPointerSize(Model.Architecture);
    ModelType.UnqualifiedType = Model.getPrimitiveType(Generic, PtrSize);
  } else {
    revng_abort("Only integer and pointer types can be directly converted "
                "from LLVM types to C types.");
  }

  return ModelType;
}

QualifiedType
deserializeFromLLVMString(llvm::Value *V, const model::Binary &Model) {
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
  ParsedType.UnqualifiedType.Root = &Model;
  revng_assert(ParsedType.UnqualifiedType.isValid());

  return ParsedType;
}

llvm::Constant *
serializeToLLVMString(model::QualifiedType &QT, llvm::Module &M) {
  // Create a string containing a serialization of the model type
  std::string SerializedQT;
  {
    llvm::raw_string_ostream StringStream(SerializedQT);
    llvm::yaml::Output YAMLOutput(StringStream);
    YAMLOutput << QT;
  }

  // Build a constant global string containing the serialized type
  return buildStringPtr(&M, SerializedQT, "");
}

RecursiveCoroutine<model::QualifiedType>
dropPointer(const model::QualifiedType &QT) {
  revng_assert(QT.isPointer());

  auto QEnd = QT.Qualifiers.end();
  for (auto QIt = QT.Qualifiers.begin(); QIt != QEnd; ++QIt) {

    if (model::Qualifier::isConst(*QIt))
      continue;

    if (model::Qualifier::isPointer(*QIt)) {
      rc_return model::QualifiedType(QT.UnqualifiedType,
                                     { std::next(QIt), QEnd });
    } else {
      revng_abort("Error: this is not a pointer");
    }

    rc_return QT;
  }

  // Recur if it has no pointer qualifier but it is a Typedef
  if (auto *TD = dyn_cast<model::TypedefType>(QT.UnqualifiedType.get()))
    rc_return rc_recur dropPointer(TD->UnderlyingType);

  revng_abort("Cannot dropPointer, QT does not have pointer qualifiers");

  rc_return{};
}

RecursiveCoroutine<QualifiedType>
getFieldType(const QualifiedType &Parent, uint64_t Idx) {
  if (Parent.isPointer())
    revng_abort("Cannot traverse a pointer");

  // If it's an array, we want to discard any const qualifier we have before the
  // first array qualifier, and the first array qualifier itself
  if (Parent.isArray()) {
    for (auto It = Parent.Qualifiers.begin(); It != Parent.Qualifiers.end();
         ++It) {

      switch (It->Kind) {
      case model::QualifierKind::Const:
        continue;
        break;

      case model::QualifierKind::Pointer:
        revng_abort("This is not an array");
        break;

      case model::QualifierKind::Array: {
        revng_assert(Idx < It->Size);
        auto ReturnedType = model::QualifiedType(Parent.UnqualifiedType, {});
        std::copy(std::next(It),
                  Parent.Qualifiers.end(),
                  ReturnedType.Qualifiers.begin());
        rc_return ReturnedType;
      }

      default:
        revng_abort("Unhandled Qualifier");
      }
    }

    revng_unreachable("If it's an array, we should have visited an array "
                      "qualifier");
  }

  // If we arrived here, there should be only const qualifiers left
  revng_assert(llvm::all_of(Parent.Qualifiers, Qualifier::isConst));
  auto *UnqualType = Parent.UnqualifiedType.getConst();

  // Traverse the UnqualifiedType
  if (auto *Struct = dyn_cast<model::StructType>(UnqualType)) {
    rc_return Struct->Fields.at(Idx).Type;
  } else if (auto *Union = dyn_cast<model::UnionType>(UnqualType)) {
    rc_return Union->Fields.at(Idx).Type;
  } else if (auto *Typedef = dyn_cast<model::TypedefType>(UnqualType)) {
    rc_return rc_recur getFieldType(Typedef->UnderlyingType, Idx);
  }

  revng_abort("Type does not contain fields");
}

QualifiedType getFieldType(const QualifiedType &Parent, llvm::Value *Idx) {
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

QualifiedType
traverseTypeSystem(const QualifiedType &Base,
                   const llvm::SmallVector<llvm::Value *, 8> &Indexes) {
  QualifiedType CurType = Base;
  for (auto Idx : Indexes)
    CurType = getFieldType(CurType, Idx);

  return CurType;
}

QualifiedType
traverseModelGEP(const model::Binary &Model, const llvm::CallInst *Call) {
  // Deduce the base type from the first argument
  QualifiedType CurType = deserializeFromLLVMString(Call->getArgOperand(0),
                                                    Model);

  // Traverse the model
  for (auto &CurArg : llvm::drop_begin(Call->args(), ModelGEPBaseArgIndex + 1))
    CurType = getFieldType(CurType, CurArg);

  return CurType;
}

RecursiveCoroutine<llvm::SmallVector<QualifiedType>>
getStrongModelInfo(const llvm::Instruction *Inst, const model::Binary &Model) {
  llvm::SmallVector<QualifiedType> ReturnTypes;

  auto ParentFunc = [&Model, &Inst]() {
    return llvmToModelFunction(Model, *Inst->getParent()->getParent());
  };

  if (auto *Call = dyn_cast<llvm::CallInst>(Inst)) {

    if (FunctionTags::CallToLifted.isTagOf(Call)) {
      // Isolated functions have their prototype in the model
      auto Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype.isValid());
      auto PrototypePath = Prototype.get();

      if (auto *RawPrototype = dyn_cast<RawFT>(PrototypePath)) {
        for (model::QualifiedType RetVal :
             llvm::map_range(RawPrototype->ReturnValues,
                             [](const model::TypedRegister &TR) {
                               return TR.Type;
                             }))
          ReturnTypes.push_back(std::move(RetVal));
      } else if (auto *CABIPrototype = dyn_cast<CABIFT>(PrototypePath)) {
        const auto Layout = abi::FunctionType::Layout::make(*CABIPrototype);
        // Layout has an invalid ReturnValue.Type if PrototypePath returns void.
        model::QualifiedType ReturnType = Layout.ReturnValue.Type;
        if (ReturnType.UnqualifiedType.isValid()) {
          if (ReturnType.isScalar()) {
            ReturnTypes.push_back(std::move(ReturnType));
          } else {
            ReturnType = peelConstAndTypedefs(ReturnType);
            revng_assert(ReturnType.is(model::TypeKind::StructType));
            revng_assert(ReturnType.Qualifiers.empty());
            auto *ModelReturnType = ReturnType.UnqualifiedType.get();
            auto *StructReturnType = cast<model::StructType>(ModelReturnType);
            for (model::QualifiedType FieldType :
                 llvm::map_range(StructReturnType->Fields,
                                 [](const model::StructField &F) {
                                   return F.Type;
                                 })) {
              revng_assert(FieldType.isScalar());
              ReturnTypes.push_back(std::move(FieldType));
            }
          }
        }
      } else {
        revng_abort("Unknown prototype kind.");
      }
    } else {
      // Non-isolated functions do not have a Prototype in the model, but we
      // can infer their returned type(s) in other ways
      auto *CalledFunc = Call->getCalledFunction();
      const auto &FuncName = CalledFunc->getName();
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      if (FTags.contains(FunctionTags::ModelGEP)
          or FTags.contains(FunctionTags::ModelGEPRef)) {
        auto GEPpedType = traverseModelGEP(Model, Call);
        ReturnTypes.push_back(GEPpedType);

      } else if (FTags.contains(FunctionTags::AddressOf)) {
        // The first argument is the base type (not the pointer's type)
        auto Base = deserializeFromLLVMString(Call->getArgOperand(0), Model);
        addPointerQualifier(Base, Model);

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
        auto *RawPrototype = cast<RawFT>(ParentFunc()->Prototype.get());

        for (const auto &RetVal : RawPrototype->ReturnValues)
          ReturnTypes.push_back(RetVal.Type);

      } else if (FTags.contains(FunctionTags::SegmentRef)) {
        const auto &[StartAddress,
                     VirtualSize] = extractSegmentKeyFromMetadata(*CalledFunc);
        auto Segment = Model.Segments.at({ StartAddress, VirtualSize });

        ReturnTypes.push_back(Segment.Type);
      } else if (FuncName.startswith("revng_stack_frame")) {
        // Retrieve the stack frame type
        auto &StackType = ParentFunc()->StackFrameType;
        revng_assert(StackType.get());

        ReturnTypes.push_back(QualifiedType{ StackType, {} });

      } else if (FuncName.startswith("revng_call_stack_arguments")) {
        // The prototype attached to this callsite represents the prototype of
        // the function that needs the stack arguments returned by this call
        auto Prototype = getCallSitePrototype(Model, Call, ParentFunc());
        revng_assert(Prototype.isValid());

        // Only RawFunctionTypes have explicit stack arguments
        auto *RawPrototype = cast<RawFT>(Prototype.get());
        QualifiedType StackArgsType = RawPrototype->StackArgumentsType;

        ReturnTypes.push_back(StackArgsType);
      }
    }
  } else if (auto *EV = llvm::dyn_cast<llvm::ExtractValueInst>(Inst)) {
    const llvm::Value *AggregateOp = EV->getAggregateOperand();

    // Transparently traverse markers backwards to find the original source of
    // the aggregate value
    while (auto *Call = isCallToTagged(AggregateOp, FunctionTags::Marker))
      AggregateOp = Call->getArgOperand(0);

    if (auto *OriginalInst = llvm::dyn_cast<llvm::Instruction>(AggregateOp))
      rc_return rc_recur getStrongModelInfo(OriginalInst, Model);
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
    if (FunctionTags::CallToLifted.isTagOf(Call)) {
      // Isolated functions have their prototype in the model
      auto Prototype = getCallSitePrototype(Model, Call);
      revng_assert(Prototype.isValid());
      auto PrototypePath = Prototype.get();

      // If we are inspecting the callee return the prototype
      if (Call->isCallee(U))
        return { createPointerTo(Prototype, Model) };

      if (Call->isArgOperand(U)) {
        unsigned int ArgOperandIdx = Call->getArgOperandNo(U);

        if (auto *CPrototype = dyn_cast<CABIFT>(PrototypePath)) {
          return { CPrototype->Arguments.at(ArgOperandIdx).Type };

        } else if (auto *RawPrototype = dyn_cast<RawFT>(PrototypePath)) {
          if (ArgOperandIdx < RawPrototype->Arguments.size()) {
            auto ArgIt = RawPrototype->Arguments.begin() + ArgOperandIdx;
            return { ArgIt->Type };

          } else {
            // If the LLVM argument is past the last model argument, we are
            // inspecting the StackArgument of the call
            revng_assert(ArgOperandIdx == RawPrototype->Arguments.size());

            auto StackArgs = RawPrototype->StackArgumentsType;
            revng_assert(StackArgs.UnqualifiedType.isValid());

            addPointerQualifier(StackArgs, Model);
            return { StackArgs };
          }
        } else {
          revng_abort("Unknown Function Type");
        }
      }
    } else {
      // Non-isolated functions do not have a Prototype in the model, but they
      // can carry type information on their operands
      revng_assert(not Call->isIndirectCall());
      unsigned int ArgOperandIdx = Call->getArgOperandNo(U);

      auto *CalledFunc = Call->getCalledFunction();
      auto FTags = FunctionTags::TagsSet::from(CalledFunc);

      if (FTags.contains(FunctionTags::AddressOf)
          or FTags.contains(FunctionTags::ModelGEPRef)) {
        // We have model type information only for the base value
        if (ArgOperandIdx != ModelGEPBaseArgIndex)
          return {};

        // The type of the base value is contained in the first operand
        auto Base = deserializeFromLLVMString(Call->getArgOperand(0), Model);
        return { Base };

      } else if (FTags.contains(FunctionTags::ModelGEP)) {
        // We have model type information only for the base value
        if (ArgOperandIdx != ModelGEPBaseArgIndex)
          return {};

        // The pointed type is contained in the first operand
        QualifiedType Base = deserializeFromLLVMString(Call->getArgOperand(0),
                                                       Model);
        addPointerQualifier(Base, Model);
        return { std::move(Base) };

      } else if (FTags.contains(FunctionTags::StructInitializer)) {
        // Struct initializers are only used to pack together return values of
        // RawFunctionTypes that return multiple values, therefore they have
        // the same type as the parent function's return type
        revng_assert(Call->getFunction()->getReturnType() == Call->getType());
        auto *RawPrototype = cast<RawFT>(ParentFunc()->Prototype.get());

        auto ArgIt = RawPrototype->ReturnValues.begin() + ArgOperandIdx;
        return { ArgIt->Type };
      }
    }
  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(User)) {
    auto ParentPrototype = ParentFunc()->Prototype.getConst();

    if (auto *RawProto = dyn_cast<RawFT>(ParentPrototype)) {
      llvm::SmallVector<QualifiedType> ReturnTypes;

      for (auto RetVal : RawProto->ReturnValues)
        ReturnTypes.push_back(RetVal.Type);

      return ReturnTypes;

    } else if (auto *CABIProto = dyn_cast<CABIFT>(ParentPrototype)) {
      return { CABIProto->ReturnType };

    } else {
      revng_abort("Unknown prototype type");
    }
  }

  return {};
}
