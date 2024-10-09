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
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/Annotations.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using llvm::dyn_cast;
using llvm::StringRef;
using llvm::Twine;
using tokenDefinition::types::TypeString;

using ptml::Tag;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

using namespace ArtificialTypes;

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
static std::string getArgumentLocation(llvm::StringRef ArgumentName,
                                       const FunctionType &F,
                                       ptml::CBuilder &B) {
  return B.getTag(ptml::tags::Span, ArgumentName)
    .addAttribute(attributes::Token, tokens::FunctionParameter)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  toStringVariableLocation(ArgumentName, F))
    .toString();
}

static std::string
getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                              const model::DynamicFunction &F,
                              ptml::CBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

static std::string getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                                                 const model::Function &F,
                                                 ptml::CBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

std::string getArgumentLocationReference(llvm::StringRef ArgumentName,
                                         const model::Function &F,
                                         ptml::CBuilder &B) {
  return getArgumentLocation<false>(ArgumentName, F, B);
}

template<bool IsDefinition>
static std::string getVariableLocation(llvm::StringRef VariableName,
                                       const model::Function &F,
                                       ptml::CBuilder &B) {
  return B.getTag(ptml::tags::Span, VariableName)
    .addAttribute(attributes::Token, tokens::Variable)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  toStringVariableLocation(VariableName, F))
    .toString();
}

std::string getVariableLocationDefinition(llvm::StringRef VariableName,
                                          const model::Function &F,
                                          ptml::CBuilder &B) {
  return getVariableLocation<true>(VariableName, F, B);
}

std::string getVariableLocationReference(llvm::StringRef VariableName,
                                         const model::Function &F,
                                         ptml::CBuilder &B) {
  return getVariableLocation<false>(VariableName, F, B);
}

struct NamedCInstanceImpl {
  const ptml::CBuilder &B;
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

TypeString getNamedCInstance(const model::Type &Type,
                             StringRef InstanceName,
                             const ptml::CBuilder &B,
                             llvm::ArrayRef<std::string> AllowedActions,
                             bool OmitInnerTypeName) {
  NamedCInstanceImpl Helper(B, AllowedActions, OmitInnerTypeName);

  std::string Result = InstanceName.str();
  Result = Helper.getString(Type, std::move(Result));

  return TypeString(std::move(Result));
}

static RecursiveCoroutine<std::string>
getArrayWrapperImpl(const model::Type &Type, const ptml::CBuilder &B) {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    std::string Result = "array_" + std::to_string(Array->ElementCount())
                         + "_of_";
    Result += rc_recur getArrayWrapperImpl(*Array->ElementType(), B);
    rc_return Result;

  } else if (auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += D->unwrap().name().str().str());

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_ptr_to_" : "ptr_to_");
    rc_return std::move(Result += rc_recur
                          getArrayWrapperImpl(*Pointer->PointeeType(), B));

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return std::move(Result += Primitive->getCName());

  } else {
    revng_abort("Unsupported model::Type.");
  }
}

TypeString getArrayWrapper(const model::ArrayType &ArrayType,
                           const ptml::CBuilder &B) {
  std::string Result = ArrayWrapperPrefix;

  Result += getArrayWrapperImpl(ArrayType, B);

  return TypeString(B.getTag(ptml::tags::Span, std::move(Result)).toString());
}

TypeString getNamedInstanceOfReturnType(const model::TypeDefinition &Function,
                                        llvm::StringRef InstanceName,
                                        const ptml::CBuilder &B,
                                        bool IsDefinition) {
  TypeString Result;
  std::vector<std::string> AllowedActions = { ptml::actions::Rename };

  using namespace abi::FunctionType;
  const auto Layout = Layout::make(Function);

  auto ReturnMethod = Layout.returnMethod();

  switch (ReturnMethod) {
  case abi::FunctionType::ReturnMethod::Void:
    Result = B.getTag(ptml::tags::Span, "void")
               .addAttribute(attributes::Token, tokens::Type)
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
      Result = getArrayWrapper(*Array, B);
      if (not InstanceName.empty())
        Result.append((Twine(" ") + Twine(InstanceName)).str());

    } else {
      Result = getNamedCInstance(*ReturnType, InstanceName, B, AllowedActions);
    }

  } break;

  case ReturnMethod::RegisterSet: {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    revng_assert(llvm::isa<model::RawFunctionDefinition>(Function));
    std::string Name = (Twine(RetStructPrefix) + Function.name()).str();
    std::string Location = pipeline::locationString(ranks::ArtificialStruct,
                                                    Function.key());
    Result = B.tokenTag(Name, ptml::c::tokens::Type)
               .addAttribute(B.getLocationAttribute(IsDefinition), Location)
               .toString();
    if (not InstanceName.empty())
      Result.append((Twine(" ") + Twine(InstanceName)).str());
  } break;

  default:
    revng_abort();
  }

  revng_assert(not llvm::StringRef(Result).trim().empty());
  return TypeString(B.getTag(ptml::tags::Span, Result)
                      .addAttribute(attributes::ActionContextLocation,
                                    locationString(ranks::ReturnValue,
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
static void printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::RawFunctionDefinition &RF,
                                       const llvm::StringRef &FunctionName,
                                       llvm::raw_ostream &Header,
                                       ptml::CBuilder &B,
                                       const model::Binary &Model,
                                       bool SingleLine) {
  using namespace abi::FunctionType;
  auto Layout = Layout::make(RF);
  revng_assert(not Layout.hasSPTAR());
  revng_assert(Layout.returnMethod() != ReturnMethod::ModelAggregate);

  auto ABI = model::Architecture::getName(RF.Architecture());
  Header << ptml::AttributeRegistry::getAnnotation<"_ABI">("raw_" + ABI.str());
  if (Function and not Function->Attributes().empty())
    Header << getFunctionAttributesString(Function->Attributes());
  Header << (SingleLine ? " " : "\n");
  Header << getNamedInstanceOfReturnType(RF, FunctionName, B, false);

  if (RF.Arguments().empty() and RF.StackArgumentsType().isEmpty()) {
    Header << "(" << B.tokenTag("void", ptml::c::tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;
    for (const model::NamedTypedRegister &Arg : RF.Arguments()) {
      std::string ArgName = Arg.name().str().str();
      std::string ArgString;
      if (Function != nullptr)
        ArgString = getArgumentLocationDefinition(ArgName, *Function, B);

      std::string
        MarkedType = getNamedCInstance(*Arg.Type(), ArgString, B).str().str();
      auto Name = model::Register::getName(Arg.Location());
      std::string Reg = ptml::AttributeRegistry::getAnnotation<"_REG">(Name);
      Tag ArgTag = B.getTag(ptml::tags::Span, MarkedType + " " + Reg);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          locationString(ranks::RawArgument,
                                         RF.key(),
                                         Arg.key()));

      Header << Separator << ArgTag.toString();
      Separator = Comma;
    }

    if (not RF.StackArgumentsType().isEmpty()) {
      // Add last argument representing a pointer to the stack arguments
      std::string StackArgName;
      if (Function != nullptr)
        StackArgName = getArgumentLocationDefinition("_stack_arguments",
                                                     *Function,
                                                     B);
      Header << Separator
             << getNamedCInstance(*RF.StackArgumentsType(), StackArgName, B);
      Header << " " << ptml::AttributeRegistry::getAttribute<"_STACK">();
    }
    Header << ")";
  }
}

template<ModelFunction FunctionType>
static void printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::CABIFunctionDefinition &CF,
                                       const llvm::StringRef &FunctionName,
                                       llvm::raw_ostream &Header,
                                       ptml::CBuilder &B,
                                       const model::Binary &Model,
                                       bool SingleLine) {
  std::string_view ABIName = model::ABI::getName(CF.ABI());
  Header << ptml::AttributeRegistry::getAnnotation<"_ABI">(ABIName);
  if (Function and not Function->Attributes().empty())
    Header << getFunctionAttributesString(Function->Attributes());
  Header << (SingleLine ? " " : "\n");
  Header << getNamedInstanceOfReturnType(CF, FunctionName, B, false);

  if (CF.Arguments().empty()) {
    Header << "(" << B.tokenTag("void", ptml::c::tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Arg : CF.Arguments()) {
      std::string ArgName = Arg.name().str().str();
      std::string ArgString;
      if (Function != nullptr)
        ArgString = getArgumentLocationDefinition(ArgName, *Function, B);

      TypeString ArgDeclaration;
      if (const model::ArrayType *Array = Arg.Type()->getArray()) {
        ArgDeclaration = getArrayWrapper(*Array, B);
        if (not ArgString.empty()) {
          ArgDeclaration.append(" ");
          ArgDeclaration.append(ArgString);
        }
      } else {
        ArgDeclaration = getNamedCInstance(*Arg.Type(), ArgString, B);
      }

      Tag ArgTag = B.getTag(ptml::tags::Span, ArgDeclaration);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          locationString(ranks::CABIArgument,
                                         CF.key(),
                                         Arg.key()));
      Header << Separator << ArgTag.toString();
      Separator = Comma;
    }
    Header << ")";
  }
}

void printFunctionPrototype(const model::TypeDefinition &FT,
                            const model::Function &Function,
                            llvm::raw_ostream &Header,
                            ptml::CBuilder &B,
                            const model::Binary &Model,
                            bool SingleLine) {
  std::string Location = locationString(ranks::Function, Function.key());
  Tag FunctionTag = B.tokenTag(Function.name(), ptml::c::tokens::Function)
                      .addAttribute(attributes::ActionContextLocation, Location)
                      .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.toString(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.toString(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else {
    revng_abort();
  }
}

void printFunctionPrototype(const model::TypeDefinition &FT,
                            const model::DynamicFunction &Function,
                            llvm::raw_ostream &Header,
                            ptml::CBuilder &B,
                            const model::Binary &Model,
                            bool SingleLine) {
  std::string Location = locationString(ranks::DynamicFunction, Function.key());
  Tag FunctionTag = B.tokenTag(Function.name(), ptml::c::tokens::Function)
                      .addAttribute(attributes::ActionContextLocation, Location)
                      .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.toString(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.toString(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else {
    revng_abort();
  }
}

void printFunctionTypeDeclaration(const model::TypeDefinition &FT,
                                  llvm::raw_ostream &Header,
                                  ptml::CBuilder &B,
                                  const model::Binary &Model) {

  auto TypeName = B.getLocationDefinition(FT);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl<model::Function>(nullptr,
                                                *RF,
                                                TypeName,
                                                Header,
                                                B,
                                                Model,
                                                true);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl<model::Function>(nullptr,
                                                *CF,
                                                TypeName,
                                                Header,
                                                B,
                                                Model,
                                                true);
  } else {
    revng_abort();
  }
}
