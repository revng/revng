//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/Annotations.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/PTMLCTypeBuilder.h"

using llvm::dyn_cast;
using llvm::StringRef;
using llvm::Twine;
using tokenDefinition::types::TypeString;

using ptml::Tag;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

template<typename FT>
concept ModelFunction = std::same_as<FT, model::Function>
                        or std::same_as<FT, model::DynamicFunction>;

static std::string toStringVariableLocation(llvm::StringRef VariableName,
                                            const model::DynamicFunction &F) {
  return pipeline::locationString(ranks::DynamicFunctionArgument,
                                  F.key(),
                                  VariableName.str());
}

static std::string toStringVariableLocation(llvm::StringRef VariableName,
                                            const model::Function &F) {
  return pipeline::locationString(ranks::LocalVariable,
                                  F.key(),
                                  VariableName.str());
}

template<bool IsDefinition, ModelFunction FunctionType>
std::string getArgumentLocation(llvm::StringRef ArgumentName,
                                const FunctionType &F,
                                const ptml::CTypeBuilder &B) {
  return B.getTag(ptml::tags::Span, ArgumentName)
    .addAttribute(attributes::Token, tokens::FunctionParameter)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  toStringVariableLocation(ArgumentName, F))
    .toString();
}

static std::string
getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                              const model::DynamicFunction &F,
                              const ptml::CTypeBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

static std::string getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                                                 const model::Function &F,
                                                 const ptml::CTypeBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

using PCTB = ptml::CTypeBuilder;
std::string PCTB::getArgumentLocationReference(llvm::StringRef ArgumentName,
                                               const model::Function &F) const {
  return getArgumentLocation<false>(ArgumentName, F, *this);
}

template<bool IsDefinition>
static std::string getVariableLocation(llvm::StringRef VariableName,
                                       const model::Function &F,
                                       const ptml::CTypeBuilder &B) {
  return B.getTag(ptml::tags::Span, VariableName)
    .addAttribute(attributes::Token, tokens::Variable)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  toStringVariableLocation(VariableName, F))
    .toString();
}

std::string
PCTB::getVariableLocationDefinition(llvm::StringRef Name,
                                    const model::Function &F) const {
  return getVariableLocation<true>(Name, F, *this);
}

std::string PCTB::getVariableLocationReference(llvm::StringRef Name,
                                               const model::Function &F) const {
  return getVariableLocation<false>(Name, F, *this);
}

struct NamedCInstanceImpl {
  ptml::CTypeBuilder &B;
  llvm::ArrayRef<std::string> AllowedActions;
  bool OmitInnerTypeName;

public:
  RecursiveCoroutine<std::string> getString(const model::Type &Type,
                                            std::string &&Emitted,
                                            bool PreviousWasAPointer = false) {
    bool NeedsSpace = true; // Emit a space except in cases where we are
    if (Emitted.empty())
      NeedsSpace = false; // emitting a nameless instance,
    if (llvm::isa<model::PointerType>(Type) and not Type.IsConst())
      NeedsSpace = false; // a non-const pointer,
    if (llvm::isa<model::ArrayType>(Type))
      NeedsSpace = false; // or an array.

    if (NeedsSpace)
      Emitted = " " + std::move(Emitted);

    if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
      rc_return rc_recur impl(*Array, std::move(Emitted), PreviousWasAPointer);

    } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
      rc_return rc_recur impl(*Pointer,
                              std::move(Emitted),
                              PreviousWasAPointer);

    } else if (auto *Def = llvm::dyn_cast<model::DefinedType>(&Type)) {
      rc_return rc_recur impl(*Def, std::move(Emitted));

    } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
      rc_return rc_recur impl(*Primitive, std::move(Emitted));

    } else {
      revng_abort("Unsupported type.");
    }
  }

private:
  RecursiveCoroutine<std::string> impl(const model::ArrayType &Array,
                                       std::string &&Emitted,
                                       bool PreviousWasAPointer) {
    revng_assert(Array.IsConst() == false);

    if (PreviousWasAPointer)
      Emitted = "(" + std::move(Emitted) + ")";

    Emitted += "[" + std::to_string(Array.ElementCount()) + "]";
    rc_return rc_recur getString(*Array.ElementType(),
                                 std::move(Emitted),
                                 false);
  }

  RecursiveCoroutine<std::string> impl(const model::PointerType &Pointer,
                                       std::string &&Emitted,
                                       bool PreviousWasAPointer) {
    std::string Current = B.getTag(ptml::tags::Span, "*")
                            .addAttribute(attributes::Token, tokens::Operator)
                            .toString();
    if (Pointer.IsConst())
      Current += constKeyword();
    Current += std::move(Emitted);

    rc_return rc_recur getString(*Pointer.PointeeType(),
                                 std::move(Current),
                                 true);
  }

  RecursiveCoroutine<std::string> impl(const model::DefinedType &Def,
                                       std::string &&Emitted) {
    std::string Result = "";
    if (not OmitInnerTypeName) {
      if (Def.IsConst())
        Result += constKeyword() + " ";

      Result += B.getLocationReference(Def.unwrap(), AllowedActions);
    }

    Result += std::move(Emitted);

    rc_return Result;
  }

  RecursiveCoroutine<std::string> impl(const model::PrimitiveType &Primitive,
                                       std::string &&Emitted) {
    std::string Result = "";
    if (not OmitInnerTypeName) {
      if (Primitive.IsConst())
        Result += constKeyword() + " ";

      Result += B.getLocationReference(Primitive);
    }

    Result += std::move(Emitted);

    rc_return Result;
  }

  std::string constKeyword() {
    return B.getKeyword(ptml::CBuilder::Keyword::Const).toString();
  }
};

TypeString PCTB::getNamedCInstance(const model::Type &Type,
                                   StringRef InstanceName,
                                   llvm::ArrayRef<std::string> AllowedActions,
                                   bool OmitInnerTypeName) {
  NamedCInstanceImpl Helper(*this, AllowedActions, OmitInnerTypeName);

  std::string Result = InstanceName.str();
  Result = Helper.getString(Type, std::move(Result));

  return TypeString(std::move(Result));
}

TypeString PCTB::getArrayWrapper(const model::ArrayType &ArrayType) {
  auto Name = NameBuilder.artificialArrayWrapperName(ArrayType);
  return TypeString(getTag(ptml::tags::Span, std::move(Name)).toString());
}

TypeString
PCTB::getNamedInstanceOfReturnType(const model::TypeDefinition &Function,
                                   llvm::StringRef InstanceName,
                                   bool IsDefinition) {
  TypeString Result;
  std::vector<std::string> AllowedActions = { ptml::actions::Rename };

  using namespace abi::FunctionType;
  const auto Layout = Layout::make(Function);

  auto ReturnMethod = Layout.returnMethod();

  switch (ReturnMethod) {
  case abi::FunctionType::ReturnMethod::Void:
    Result = getTag(ptml::tags::Span, "void")
               .addAttribute(attributes::Token, c::tokens::Type)
               .toString();
    if (not InstanceName.empty())
      Result.append((Twine(" ") + Twine(InstanceName)).str());
    break;

  case ReturnMethod::ModelAggregate:
  case ReturnMethod::Scalar: {
    const model::Type *ReturnType = nullptr;

    if (ReturnMethod == ReturnMethod::ModelAggregate) {
      ReturnType = &Layout.returnValueAggregateType();
    } else {
      revng_assert(Layout.ReturnValues.size() == 1);
      ReturnType = Layout.ReturnValues[0].Type.get();
    }

    // When returning arrays, they need to be wrapped into an artificial
    // struct
    if (const model::ArrayType *Array = ReturnType->getArray()) {
      Result = getArrayWrapper(*Array);
      if (not InstanceName.empty())
        Result.append((Twine(" ") + Twine(InstanceName)).str());

    } else {
      Result = getNamedCInstance(*ReturnType, InstanceName, AllowedActions);
    }

  } break;

  case ReturnMethod::RegisterSet: {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    auto *RFT = llvm::dyn_cast<model::RawFunctionDefinition>(&Function);
    revng_assert(RFT);

    std::string Location = pipeline::locationString(ranks::ArtificialStruct,
                                                    RFT->key());
    Result = tokenTag(NameBuilder.artificialReturnValueWrapperName(*RFT),
                      ptml::c::tokens::Type)
               .addAttribute(getLocationAttribute(IsDefinition), Location)
               .toString();
    if (not InstanceName.empty())
      Result.append((Twine(" ") + Twine(InstanceName)).str());
  } break;

  default:
    revng_abort();
  }

  revng_assert(not llvm::StringRef(Result).trim().empty());
  return TypeString(getTag(ptml::tags::Span, Result)
                      .addAttribute(attributes::ActionContextLocation,
                                    pipeline::locationString(ranks::ReturnValue,
                                                             Function.key()))
                      .toString());
}

static std::string
getFunctionAttributeString(const model::FunctionAttribute::Values &A) {

  using namespace model::FunctionAttribute;

  switch (A) {

  case NoReturn:
    return "_Noreturn";

  case Inline:
    return "inline";

  default:
    revng_abort("cannot print unexpected model::FunctionAttribute");
  }

  return "";
}

using AttributesSet = TrackingMutableSet<model::FunctionAttribute::Values>;

static std::string
getFunctionAttributesString(const AttributesSet &Attributes) {
  std::string Result;
  for (const auto &A : Attributes)
    Result += " " + getFunctionAttributeString(A);
  return Result;
}

template<ModelFunction FunctionType>
std::string printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::RawFunctionDefinition &RF,
                                       const llvm::StringRef &FunctionName,
                                       ptml::CTypeBuilder &B,
                                       bool SingleLine) {
  using namespace abi::FunctionType;
  auto Layout = Layout::make(RF);
  revng_assert(not Layout.hasSPTAR());
  revng_assert(Layout.returnMethod() != ReturnMethod::ModelAggregate);

  std::string Result;
  auto ABI = model::Architecture::getName(RF.Architecture());
  Result += ptml::AttributeRegistry::getAnnotation<"_ABI">("raw_" + ABI.str());
  if (Function and not Function->Attributes().empty())
    Result += getFunctionAttributesString(Function->Attributes());
  Result += (SingleLine ? " " : "\n");
  Result += B.getNamedInstanceOfReturnType(RF, FunctionName, false);

  if (RF.Arguments().empty() and RF.StackArgumentsType().isEmpty()) {
    Result += "(" + B.tokenTag("void", ptml::c::tokens::Type) + ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;
    for (const model::NamedTypedRegister &Arg : RF.Arguments()) {
      std::string ArgName = B.NameBuilder.argumentName(RF, Arg).str().str();
      std::string ArgString;
      if (Function != nullptr)
        ArgString = getArgumentLocationDefinition(ArgName, *Function, B);

      std::string
        MarkedType = B.getNamedCInstance(*Arg.Type(), ArgString).str().str();
      auto Name = model::Register::getName(Arg.Location());
      std::string Reg = ptml::AttributeRegistry::getAnnotation<"_REG">(Name);
      Tag ArgTag = B.getTag(ptml::tags::Span, MarkedType + " " + Reg);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          locationString(ranks::RawArgument,
                                         RF.key(),
                                         Arg.key()));

      Result += Separator.str() + ArgTag.toString();
      Separator = Comma;
    }

    if (not RF.StackArgumentsType().isEmpty()) {
      // Add last argument representing a pointer to the stack arguments
      std::string StackArgName;
      if (Function != nullptr)
        StackArgName = getArgumentLocationDefinition("_stack_arguments",
                                                     *Function,
                                                     B);
      auto N = B.getNamedCInstance(*RF.StackArgumentsType(), StackArgName);
      Result += Separator.str() + N.str().str() + " "
                + ptml::AttributeRegistry::getAttribute<"_STACK">();
    }
    Result += ")";
  }

  return Result;
}

template<ModelFunction FunctionType>
std::string printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::CABIFunctionDefinition &CF,
                                       const llvm::StringRef &FunctionName,
                                       ptml::CTypeBuilder &B,
                                       bool SingleLine) {
  std::string Result;

  std::string_view ABIName = model::ABI::getName(CF.ABI());
  Result += ptml::AttributeRegistry::getAnnotation<"_ABI">(ABIName);
  if (Function and not Function->Attributes().empty())
    Result += getFunctionAttributesString(Function->Attributes());
  Result += (SingleLine ? " " : "\n");
  Result += B.getNamedInstanceOfReturnType(CF, FunctionName, false);

  if (CF.Arguments().empty()) {
    Result += "(" + B.tokenTag("void", ptml::c::tokens::Type) + ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Arg : CF.Arguments()) {
      std::string ArgName = B.NameBuilder.argumentName(CF, Arg).str().str();
      std::string ArgString;
      if (Function != nullptr)
        ArgString = getArgumentLocationDefinition(ArgName, *Function, B);

      TypeString ArgDeclaration;
      if (const model::ArrayType *Array = Arg.Type()->getArray()) {
        ArgDeclaration = B.getArrayWrapper(*Array);
        if (not ArgString.empty()) {
          ArgDeclaration.append(" ");
          ArgDeclaration.append(ArgString);
        }
      } else {
        ArgDeclaration = B.getNamedCInstance(*Arg.Type(), ArgString);
      }

      Tag ArgTag = B.getTag(ptml::tags::Span, ArgDeclaration);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          locationString(ranks::CABIArgument,
                                         CF.key(),
                                         Arg.key()));
      Result += Separator.str() + ArgTag.toString();
      Separator = Comma;
    }
    Result += ")";
  }

  return Result;
}

void ptml::CTypeBuilder::printFunctionPrototype(const model::TypeDefinition &FT,
                                                const model::Function &Function,
                                                bool SingleLine) {
  std::string Location = pipeline::locationString(ranks::Function,
                                                  Function.key());
  auto FunctionTag = tokenTag(NameBuilder.name(Function),
                              ptml::c::tokens::Function)
                       .addAttribute(attributes::ActionContextLocation,
                                     Location)
                       .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl(&Function,
                                       *RF,
                                       FunctionTag.toString(),
                                       *this,
                                       SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl(&Function,
                                       *CF,
                                       FunctionTag.toString(),
                                       *this,
                                       SingleLine);
  } else {
    revng_abort();
  }
}

void ptml::CTypeBuilder::printFunctionPrototype(const model::TypeDefinition &FT,
                                                const model::DynamicFunction
                                                  &Function,
                                                bool SingleLine) {
  std::string Location = pipeline::locationString(ranks::DynamicFunction,
                                                  Function.key());
  auto FunctionTag = tokenTag(NameBuilder.name(Function),
                              ptml::c::tokens::Function)
                       .addAttribute(attributes::ActionContextLocation,
                                     Location)
                       .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl(&Function,
                                       *RF,
                                       FunctionTag.toString(),
                                       *this,
                                       SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl(&Function,
                                       *CF,
                                       FunctionTag.toString(),
                                       *this,
                                       SingleLine);
  } else {
    revng_abort();
  }
}

void ptml::CTypeBuilder::printFunctionPrototype(const model::TypeDefinition
                                                  &FT) {

  auto TypeName = getLocationDefinition(FT);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl<model::Function>(nullptr,
                                                        *RF,
                                                        TypeName,
                                                        *this,
                                                        true);

  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    *Out << printFunctionPrototypeImpl<model::Function>(nullptr,
                                                        *CF,
                                                        TypeName,
                                                        *this,
                                                        true);
  } else {
    revng_abort();
  }
}

void ptml::CTypeBuilder::printSegmentType(const model::Segment &Segment) {
  if (not Segment.Type().isEmpty()) {
    *Out << getNamedCInstance(*Segment.Type(), getLocationDefinition(Segment))
         << ";\n";

  } else {
    // If the segment has no type, emit it as an array of bytes.
    auto Array = model::ArrayType::make(model::PrimitiveType::makeGeneric(1),
                                        Segment.VirtualSize());
    *Out << getNamedCInstance(*Array, getLocationDefinition(Segment)) << ";\n";
  }
}
