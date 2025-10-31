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
#include "llvm/Support/Casting.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/PTML/CBuilder.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/Annotations.h"
#include "revng/Support/Assert.h"
#include "revng/TypeNames/LLVMTypeNames.h"
#include "revng/TypeNames/ModelCBuilder.h"

using llvm::dyn_cast;
using llvm::StringRef;
using llvm::Twine;

using ptml::Tag;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

struct NamedCInstanceImpl {
  const ptml::ModelCBuilder &B;
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
    if (uint64_t Size = B.Configuration.ExplicitTargetPointerSize) {
      std::string Current;
      if (Pointer.IsConst()) {
        Current += constKeyword();
        Current += " ";
      }
      Current += "pointer";
      Current += std::to_string(Size * 8);
      Current += "_t(";
      Current += rc_recur getString(*Pointer.PointeeType(), {});
      Current += ") ";
      Current += std::move(Emitted);
      rc_return Current;
    } else {
      auto Current = B.getOperator(ptml::CBuilder::Operator::PointerDereference)
                       .toString();
      if (Pointer.IsConst())
        Current += constKeyword();
      Current += std::move(Emitted);

      rc_return rc_recur getString(*Pointer.PointeeType(),
                                   std::move(Current),
                                   true);
    }
  }

  RecursiveCoroutine<std::string> impl(const model::DefinedType &Def,
                                       std::string &&Emitted) {
    std::string Result;
    if (not OmitInnerTypeName) {
      if (Def.IsConst())
        Result += constKeyword() + " ";

      Result += B.getReferenceTag(Def.unwrap());
    }

    Result += std::move(Emitted);

    rc_return Result;
  }

  RecursiveCoroutine<std::string> impl(const model::PrimitiveType &Primitive,
                                       std::string &&Emitted) {
    std::string Result;
    if (not OmitInnerTypeName) {
      if (Primitive.IsConst())
        Result += constKeyword() + " ";

      Result += B.getReferenceTag(Primitive);
    }

    Result += std::move(Emitted);

    rc_return Result;
  }

  std::string constKeyword() {
    return B.getKeyword(ptml::CBuilder::Keyword::Const).toString();
  }
};

using PCTB = ptml::ModelCBuilder;
std::string PCTB::getNamedCInstance(const model::Type &Type,
                                    StringRef InstanceName,
                                    bool OmitInnerTypeName) const {
  NamedCInstanceImpl Helper(*this, OmitInnerTypeName);

  return Helper.getString(Type, InstanceName.str());
}

std::string
PCTB::getNamedCInstanceOfReturnType(const model::TypeDefinition &Function,
                                    llvm::StringRef InstanceName) const {
  std::string Suffix;
  if (not InstanceName.empty())
    Suffix.append(" " + InstanceName.str());

  const auto Layout = abi::FunctionType::Layout::make(Function);
  auto ReturnMethod = Layout.returnMethod();

  switch (ReturnMethod) {
  case abi::FunctionType::ReturnMethod::Void:
    return getVoidTag() + Suffix;

  case abi::FunctionType::ReturnMethod::ModelAggregate:
  case abi::FunctionType::ReturnMethod::Scalar: {
    const model::Type *ReturnType = nullptr;

    if (ReturnMethod == abi::FunctionType::ReturnMethod::ModelAggregate) {
      ReturnType = &Layout.returnValueAggregateType();
    } else {
      revng_assert(Layout.ReturnValues.size() == 1);
      ReturnType = Layout.ReturnValues[0].Type.get();
    }

    return getNamedCInstance(*ReturnType, InstanceName);
  }

  case abi::FunctionType::ReturnMethod::RegisterSet: {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    const auto &RFT = llvm::cast<model::RawFunctionDefinition>(Function);
    return getArtificialStructTag<false>(RFT) + Suffix;
  }

  default:
    revng_abort("Unsupported function return method.");
  }
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

template<typename FT>
concept ModelFunction = std::same_as<FT, model::Function>
                        or std::same_as<FT, model::DynamicFunction>;

static std::string
getReturnValueAndNameImpl(const model::TypeDefinition &FunctionType,
                          const llvm::StringRef &FunctionName,
                          const ptml::ModelCBuilder &B) {
  std::string Type = B.getFunctionReturnType(FunctionType);
  revng_assert(not Type.empty());

  // Workaround to ensure proper spacing around pointers.
  bool NeedsSpace = true;
  const auto Layout = abi::FunctionType::Layout::make(FunctionType);
  if (Layout.returnMethod() == abi::FunctionType::ReturnMethod::Scalar) {
    revng_assert(Layout.ReturnValues.size() == 1);

    // Not using `isPointer` because typedefs change the behavior here.
    using PointerT = model::PointerType;
    if (auto *P = llvm::dyn_cast<PointerT>(Layout.ReturnValues[0].Type.get()))
      NeedsSpace = P->IsConst();
  }

  return B.getReturnValueTag(std::move(Type), FunctionType)
         + (NeedsSpace ? " " : "") + FunctionName.str();
}

template<ModelFunction FunctionType>
std::string printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::RawFunctionDefinition &RF,
                                       const llvm::StringRef &FunctionName,
                                       const ptml::ModelCBuilder &B,
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
  Result += getReturnValueAndNameImpl(RF, FunctionName, B);

  if (RF.Arguments().empty() and RF.StackArgumentsType().isEmpty()) {
    Result += "(" + B.getVoidTag() + ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;
    for (const model::NamedTypedRegister &Argument : RF.Arguments()) {
      std::string ArgumentName;
      if (Function != nullptr)
        ArgumentName = B.getDefinitionTag(RF, Argument);

      std::string MarkedType = B.getNamedCInstance(*Argument.Type(),
                                                   ArgumentName);
      auto RegName = model::Register::getName(Argument.Location());
      std::string Reg = ptml::AttributeRegistry::getAnnotation<"_REG">(RegName);
      Result += Separator.str()
                + B.getCommentableTag(MarkedType + " " + Reg, RF, Argument);
      Separator = Comma;
    }

    if (not RF.StackArgumentsType().isEmpty()) {
      // Add last argument representing a pointer to the stack arguments
      std::string StackArgName;
      if (Function != nullptr)
        StackArgName = B.getStackArgumentDefinitionTag(RF);

      auto N = B.getNamedCInstance(*RF.StackArgumentsType(), StackArgName);
      static auto Attribute = ptml::AttributeRegistry::getAttribute<"_STACK">();
      Result += Separator.str()
                + B.getCommentableTag(N + " " + Attribute,
                                      *RF.stackArgumentsType());
    }
    Result += ")";
  }

  return Result;
}

template<ModelFunction FunctionType>
std::string printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::CABIFunctionDefinition &CF,
                                       const llvm::StringRef &FunctionName,
                                       const ptml::ModelCBuilder &B,
                                       bool SingleLine) {

  using namespace abi::FunctionType;
  auto Layout = Layout::make(CF);
  revng_assert(Layout.returnMethod() != ReturnMethod::RegisterSet);

  std::string Result;

  llvm::StringRef ABIName = model::ABI::getName(CF.ABI());
  Result += ptml::AttributeRegistry::getAnnotation<"_ABI">(ABIName);
  if (Function and not Function->Attributes().empty())
    Result += getFunctionAttributesString(Function->Attributes());
  Result += (SingleLine ? " " : "\n");
  Result += getReturnValueAndNameImpl(CF, FunctionName, B);

  if (CF.Arguments().empty()) {
    Result += "(" + B.getVoidTag() + ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Argument : CF.Arguments()) {
      std::string ArgumentName;
      if (Function != nullptr)
        ArgumentName = B.getDefinitionTag(CF, Argument);

      Result += Separator.str();
      Result += B.getCommentableTag(B.getNamedCInstance(*Argument.Type(),
                                                        ArgumentName),
                                    CF,
                                    Argument);
      Separator = Comma;
    }
    Result += ")";
  }

  return Result;
}

template<ModelFunction FunctionType>
std::string printFunctionPrototypeImpl(const model::TypeDefinition &FT,
                                       const FunctionType *Function,
                                       const llvm::StringRef &FunctionName,
                                       const ptml::ModelCBuilder &B,
                                       bool SingleLine) {
  std::string Result;
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    Result = printFunctionPrototypeImpl(Function,
                                        *RF,
                                        FunctionName,
                                        B,
                                        SingleLine);

  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    Result = printFunctionPrototypeImpl(Function,
                                        *CF,
                                        FunctionName,
                                        B,
                                        SingleLine);

  } else {
    revng_abort();
  }

  if (Function)
    return B.getCommentableTag(std::move(Result), *Function);
  else
    return B.getCommentableTag(std::move(Result), FT);
}

using MCB = ptml::ModelCBuilder;

void MCB::printFunctionPrototype(const model::TypeDefinition &FT,
                                 const model::Function &Function,
                                 bool SingleLine) {
  *Out << printFunctionPrototypeImpl(FT,
                                     &Function,
                                     getDefinitionTag(Function),
                                     *this,
                                     SingleLine);
}

void MCB::printFunctionPrototype(const model::TypeDefinition &FT,
                                 const model::DynamicFunction &Function,
                                 bool SingleLine) {
  *Out << printFunctionPrototypeImpl(FT,
                                     &Function,
                                     getDefinitionTag(Function),
                                     *this,
                                     SingleLine);
}

void ptml::ModelCBuilder::printFunctionPrototype(const model::TypeDefinition
                                                   &FT) {
  *Out << printFunctionPrototypeImpl(FT,
                                     (const model::Function *) nullptr,
                                     getDefinitionTag(FT),
                                     *this,
                                     true);
}

void ptml::ModelCBuilder::printSegmentType(const model::Segment &Segment) {
  std::string Result = "\n" + getModelCommentWithoutLeadingNewline(Segment);
  if (not Segment.Type().isEmpty()) {
    Result += getNamedCInstance(*Segment.Type(), getDefinitionTag(Segment));

  } else {
    // If the segment has no type, emit it as an array of bytes.
    auto Array = model::ArrayType::make(model::PrimitiveType::makeGeneric(1),
                                        Segment.VirtualSize());
    Result += getNamedCInstance(*Array, getDefinitionTag(Segment));
  }

  *Out << getCommentableTag(std::move(Result) + ";\n", Binary, Segment);
}

Logger VariableNamingLog("variable-naming");

ptml::ModelCBuilder::TagPair
ptml::ModelCBuilder::getVariableTags(VariableNameBuilder &VariableNameBuilder,
                                     const SortedVector<MetaAddress>
                                       &UserLocationSet) const {
  constexpr std::array Actions = { ptml::actions::Rename };
  llvm::ArrayRef<llvm::StringRef> CurrentActions = Actions;

  auto [Name, I, HasA, Warning] = VariableNameBuilder.name(UserLocationSet);
  if (not HasA) {
    // The variable is not locatable, ensure `rename` action is not allowed.
    constexpr std::array<llvm::StringRef, 0> EmptyActions = {};
    CurrentActions = EmptyActions;
  }

  if (VariableNamingLog.isEnabled()) {
    if (HasA)
      VariableNamingLog << "A locatable ";
    else
      VariableNamingLog << "A non-locatable ";

    VariableNamingLog << "variable (" << I << ") at '"
                      << addressesToString(UserLocationSet)
                      << "' received the name: '" << Name << "'" << DoLog;

    if (not Warning.empty())
      VariableNamingLog << "WARNING (in "
                        << ::toString(VariableNameBuilder.function().key())
                        << "): " << Warning << '\n';
  }

  // TODO: emit warning to users (comment, vscode squiggle, etc), instead of
  //       just logging it.

  auto Location = variableLocationString(VariableNameBuilder.function(), I);
  return TagPair{
    .Definition = getNameTagImpl<true>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       CurrentActions),
    .Reference = getNameTagImpl<false>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       CurrentActions)
  };
}

ptml::ModelCBuilder::TagPair
ptml::ModelCBuilder::getReservedVariableTags(const model::Function &Function,
                                             llvm::StringRef Name) const {
  NameBuilder.assertNameIsReserved(Name);

  constexpr std::array<llvm::StringRef, 0> Actions = {};

  std::string Location = reservedVariableLocationString(Function, Name);
  // NOTE: these should never be used as a context for any actions.

  revng_log(VariableNamingLog,
            "A non-renamable variable received the name: '" << Name << "'");

  return TagPair{
    .Definition = getNameTagImpl<true>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       Actions),
    .Reference = getNameTagImpl<false>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       Actions)
  };
}

ptml::ModelCBuilder::TagPair
ptml::ModelCBuilder::getGotoLabelTags(GotoLabelNameBuilder &LabelNameBuilder,
                                      const SortedVector<MetaAddress>
                                        &UserLocationSet) const {
  constexpr std::array Actions = { ptml::actions::Rename };
  llvm::ArrayRef<llvm::StringRef> CurrentActions = Actions;

  auto [Name, I, HasAddr, Warning] = LabelNameBuilder.name(UserLocationSet);
  if (not HasAddr) {
    // The label is not locatable, ensure `rename` action is not allowed.
    constexpr std::array<llvm::StringRef, 0> EmptyActions = {};
    CurrentActions = EmptyActions;
  }

  if (VariableNamingLog.isEnabled()) {
    if (HasAddr)
      VariableNamingLog << "A locatable ";
    else
      VariableNamingLog << "A non-locatable ";

    if (not Warning.empty())
      VariableNamingLog << "WARNING (in "
                        << ::toString(LabelNameBuilder.function().key())
                        << "): " << Warning << '\n';
  }

  // TODO: emit warning to users (comment, vscode squiggle, etc), instead of
  //       just logging it.

  auto Location = gotoLabelLocationString(LabelNameBuilder.function(), I);
  return TagPair{
    .Definition = getNameTagImpl<true>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       CurrentActions),
    .Reference = getNameTagImpl<false>(tokenTag(Name,
                                                ptml::c::tokens::Variable),
                                       Location,
                                       CurrentActions)
  };
}
