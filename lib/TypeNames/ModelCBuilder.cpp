//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/PTML/CAttributes.h"
#include "revng/TypeNames/ModelCBuilder.h"

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
                                    llvm::StringRef InstanceName,
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

  case AlwaysInline:
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
  Result += ptml::AttributeRegistry::getAnnotationString<"_ABI">("raw_"
                                                                 + ABI.str());
  if (Function and not Function->Attributes().empty())
    Result += getFunctionAttributesString(Function->Attributes());
  Result += (SingleLine ? " " : "\n");
  Result += getReturnValueAndNameImpl(RF, FunctionName, B);

  if (RF.Arguments().empty() and RF.StackArgumentsType().isEmpty()) {
    Result += "(" + B.getVoidTag() + ")";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;
    for (const model::NamedTypedRegister &Argument : RF.Arguments()) {
      std::string ArgumentName;
      if (Function != nullptr)
        ArgumentName = B.getDefinitionTag(RF, Argument);

      std::string MarkedType = B.getNamedCInstance(*Argument.Type(),
                                                   ArgumentName);
      auto RegName = model::Register::getName(Argument.Location());
      std::string
        Reg = ptml::AttributeRegistry::getAnnotationString<"_REG">(RegName);
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
      static auto
        Attribute = ptml::AttributeRegistry::getAttributeString<"_STACK">();
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
  Result += ptml::AttributeRegistry::getAnnotationString<"_ABI">(ABIName);
  if (Function and not Function->Attributes().empty())
    Result += getFunctionAttributesString(Function->Attributes());
  Result += (SingleLine ? " " : "\n");
  Result += getReturnValueAndNameImpl(CF, FunctionName, B);

  if (CF.Arguments().empty()) {
    Result += "(" + B.getVoidTag() + ")";
  } else {
    const llvm::StringRef Open = "(";
    const llvm::StringRef Comma = ", ";
    llvm::StringRef Separator = Open;

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
  if (auto *RF = llvm::dyn_cast<model::RawFunctionDefinition>(&FT)) {
    Result = printFunctionPrototypeImpl(Function,
                                        *RF,
                                        FunctionName,
                                        B,
                                        SingleLine);

  } else if (auto *CF = llvm::dyn_cast<model::CABIFunctionDefinition>(&FT)) {
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

void ptml::ModelCBuilder::printPadding(uint64_t FieldOffset,
                                       uint64_t NextOffset) {
  if (Configuration.EnableExplicitPadding) {
    revng_assert(FieldOffset <= NextOffset);
    if (FieldOffset == NextOffset)
      return; // There is no padding

    *Out << tokenTag("uint8_t", ptml::c::tokens::Type) << " "
         << tokenTag(NameBuilder.paddingFieldName(FieldOffset),
                     ptml::c::tokens::Field)
         << "[" << getNumber(NextOffset - FieldOffset) << "];\n";
  }
}

void ptml::ModelCBuilder::printOpaqueTypeDefinition(uint64_t ByteSize) {

  // Print the typedef inline with the struct definition.
  std::string
    StructLine = getKeyword(ptml::CBuilder::Keyword::Typedef) + " "
                 + getKeyword(ptml::CBuilder::Keyword::Struct) + " "
                 + ptml::Attributes.getAttributeString<"_PACKED">() + " "
                 + getOpaqueTypeDeclarationTag</*IsDefinition*/ false>(ByteSize)
                 + " ";
  *Out << std::move(StructLine);
  {
    Scope Scope(*Out, ptml::c::scopes::StructBody);

    // We print padding even when Configuration.EnableExplicitPadding == false
    // because otherwise the struct is empty and is not valid C99 (empty structs
    // are a language extension).
    *Out << tokenTag("uint8_t", ptml::c::tokens::Type) << " "
         << tokenTag(NameBuilder.paddingFieldName(0), ptml::c::tokens::Field)
         << "[" << getNumber(ByteSize) << "];\n";
  }

  *Out << " " << getOpaqueTypeDeclarationTag</*IsDefinition*/ true>(ByteSize)
       << ";\n";
}

std::set<uint64_t> ptml::ModelCBuilder::getModelOpaqueByteSizes() {

  std::set<uint64_t> ByteSizes = { 1, 2, 4, 8, 10, 12, 16 };

  for (const model::UpcastableTypeDefinition &Type : Binary.TypeDefinitions()) {
    uint64_t ByteSize = Type->size().value_or(0);
    if (ByteSize)
      ByteSizes.insert(ByteSize);

    llvm::SmallVector<model::UpcastableType> Dependencies;

    const model::TypeDefinition *Definition = Type->tryGetAsDefinition();
    if (not Definition)
      continue;

    if (const auto *TD = llvm::dyn_cast<model::TypedefDefinition>(Definition))
      Dependencies.push_back(TD->UnderlyingType());

    if (const auto *S = Definition->getStruct())
      for (const auto &Field : S->Fields())
        Dependencies.push_back(Field.Type());

    if (const auto *U = Definition->getUnion())
      for (const auto &Field : U->Fields())
        Dependencies.push_back(Field.Type());

    if (const auto *R = Definition->getRawFunction()) {
      for (const auto &A : R->Arguments())
        Dependencies.push_back(A.Type());

      uint64_t ReturnTypeSize = 0;
      for (const auto &RV : R->ReturnValues()) {
        Dependencies.push_back(RV.Type());
        ReturnTypeSize += RV.Type()->size().value_or(0);
      }
      if (ReturnTypeSize)
        ByteSizes.insert(ReturnTypeSize);
    }

    if (const auto *C = Definition->getCABIFunction()) {
      for (const auto &A : C->Arguments())
        Dependencies.push_back(A.Type());
      if (not C->ReturnType().isEmpty())
        Dependencies.push_back(C->ReturnType());
    }

    for (const model::UpcastableType &D : Dependencies)
      if (uint64_t ByeSize = D->trySize().value_or(0))
        ByteSizes.insert(ByteSize);
  }

  return ByteSizes;
}

void ptml::ModelCBuilder::printModelOpaqueTypeDefinitions() {
  std::set<uint64_t> ByteSizes = getModelOpaqueByteSizes();
  for (uint64_t ByteSize : ByteSizes) {
    *Out << "\n";
    printOpaqueTypeDefinition(ByteSize);
  }
}

using MCB = ptml::ModelCBuilder;
using SizeSet = std::set<uint64_t>;
void MCB::printHelperOpaqueTypeDefinitions(const SizeSet &HelperByteSizes) {
  std::set<uint64_t> ModelByteSizes = getModelOpaqueByteSizes();
  for (uint64_t ByteSize : HelperByteSizes) {
    if (not ModelByteSizes.contains(ByteSize)) {
      *Out << "\n";
      printOpaqueTypeDefinition(ByteSize);
    }
  }
}
