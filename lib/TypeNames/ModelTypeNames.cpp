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
#include "revng/Model/QualifiedType.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using llvm::dyn_cast;
using llvm::StringRef;
using llvm::Twine;
using tokenDefinition::types::TypeString;

using pipeline::serializedLocation;
using ptml::Tag;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

using namespace ArtificialTypes;

template<typename FT>
concept ModelFunction = std::same_as<FT, model::Function>
                        or std::same_as<FT, model::DynamicFunction>;

static std::string serializeVariableLocation(llvm::StringRef VariableName,
                                             const model::DynamicFunction &F) {
  return pipeline::serializedLocation(ranks::DynamicFunctionArgument,
                                      F.key(),
                                      VariableName.str());
}

static std::string serializeVariableLocation(llvm::StringRef VariableName,
                                             const model::Function &F) {
  return pipeline::serializedLocation(ranks::LocalVariable,
                                      F.key(),
                                      VariableName.str());
}

template<bool IsDefinition, ModelFunction FunctionType>
static std::string getArgumentLocation(llvm::StringRef ArgumentName,
                                       const FunctionType &F,
                                       ptml::PTMLCBuilder &B) {
  return B.getTag(ptml::tags::Span, ArgumentName)
    .addAttribute(attributes::Token, tokens::FunctionParameter)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  serializeVariableLocation(ArgumentName, F))
    .serialize();
}

static std::string
getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                              const model::DynamicFunction &F,
                              ptml::PTMLCBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

static std::string getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                                                 const model::Function &F,
                                                 ptml::PTMLCBuilder &B) {
  return getArgumentLocation<true>(ArgumentName, F, B);
}

std::string getArgumentLocationReference(llvm::StringRef ArgumentName,
                                         const model::Function &F,
                                         ptml::PTMLCBuilder &B) {
  return getArgumentLocation<false>(ArgumentName, F, B);
}

template<bool IsDefinition>
static std::string getVariableLocation(llvm::StringRef VariableName,
                                       const model::Function &F,
                                       ptml::PTMLCBuilder &B) {
  return B.getTag(ptml::tags::Span, VariableName)
    .addAttribute(attributes::Token, tokens::Variable)
    .addAttribute(B.getLocationAttribute(IsDefinition),
                  serializeVariableLocation(VariableName, F))
    .serialize();
}

std::string getVariableLocationDefinition(llvm::StringRef VariableName,
                                          const model::Function &F,
                                          ptml::PTMLCBuilder &B) {
  return getVariableLocation<true>(VariableName, F, B);
}

std::string getVariableLocationReference(llvm::StringRef VariableName,
                                         const model::Function &F,
                                         ptml::PTMLCBuilder &B) {
  return getVariableLocation<false>(VariableName, F, B);
}

TypeString getNamedCInstance(const model::QualifiedType &QT,
                             StringRef InstanceName,
                             const ptml::PTMLCBuilder &B,
                             llvm::ArrayRef<std::string> AllowedActions) {
  const model::TypeDefinition &Unqualified = *QT.UnqualifiedType().getConst();
  std::string TypeName = B.getLocationReference(Unqualified, AllowedActions);

  if (auto *Enum = dyn_cast<model::EnumDefinition>(&Unqualified)) {
    const model::QualifiedType &Underlying = Enum->UnderlyingType();
    revng_assert(Underlying.Qualifiers().empty());
    std::string UnderlyingName = B.getLocationReference(*Underlying
                                                           .UnqualifiedType()
                                                           .getConst(),
                                                        AllowedActions);

    std::string EnumTypeWithAttribute = B.getAnnotateEnum(UnderlyingName);
    EnumTypeWithAttribute += " " + std::move(TypeName);
    TypeName = std::move(EnumTypeWithAttribute);
  }

  return getNamedCInstance(TypeName, QT.Qualifiers(), InstanceName, B);
}

TypeString getNamedCInstance(StringRef TypeName,
                             const std::vector<model::Qualifier> &Qualifiers,
                             StringRef InstanceName,
                             const ptml::PTMLCBuilder &B) {
  constexpr auto &isConst = model::Qualifier::isConst;
  constexpr auto &isPointer = model::Qualifier::isPointer;

  bool IsUnqualified = Qualifiers.empty();
  bool FirstQualifierIsPointer = IsUnqualified or isPointer(Qualifiers.front());
  bool PrependWhitespaceToInstanceName = not InstanceName.empty()
                                         and (IsUnqualified
                                              or not FirstQualifierIsPointer);

  TypeString Result;

  // Here we have a bunch of pointers, const, and array qualifiers.
  // Because of arrays, we have to emit the types with C infamous clockwise
  // spiral rule. Luckily all our function types have names, so at least this
  // cannot become too nasty.

  auto QIt = Qualifiers.begin();
  auto QEnd = Qualifiers.end();
  do {
    // Accumulate the result that are outside the array.
    TypeString Partial;

    // Find the first qualifier that is an array.
    auto QArrayIt = std::find_if(QIt, QEnd, model::Qualifier::isArray);
    {
      // If we find it, go back to the first previous const-qualifier that
      // const-qualifies the array itself. This is necessary because C does not
      // have const arrays, only arrays of const, so we have to handle
      // const-arrays specially, and emit the const-qualifier on the element in
      // C, even if in the model it was on the array.
      if (QArrayIt != QEnd and QArrayIt != QIt
          and isConst(*std::make_reverse_iterator(QArrayIt)))
        QArrayIt = std::prev(QArrayIt);
    }

    // Emit non-array qualifiers.
    {
      bool PrevPointer = false;
      for (const model::Qualifier &Q :
           llvm::reverse(llvm::make_range(QIt, QArrayIt))) {
        if (not PrevPointer)
          Partial.append(" ");

        switch (Q.Kind()) {

        case model::QualifierKind::Const:
          using PTMLKW = ptml::PTMLCBuilder::Keyword;
          Partial.append(B.getKeyword(PTMLKW::Const).serialize());
          PrevPointer = false;
          break;
        case model::QualifierKind::Pointer:
          Partial.append(B.getTag(ptml::tags::Span, "*")
                           .addAttribute(attributes::Token, tokens::Operator)
                           .serialize());
          PrevPointer = true;
          break;

        default:
          revng_abort();
        }
      }
    }

    // Print the actual instance name.
    if (QIt == Qualifiers.begin()) {
      if (PrependWhitespaceToInstanceName)
        Partial.append(" ");
      Result.append(InstanceName.str());
    }

    // Now we can prepend the qualifiers that are outside the array to the
    // Result string. This always work because at this point Result holds
    // whatever is left from previous iteration, so it's either empty, or it
    // starts with '(' because we're using the clockwise spiral rule.
    Result = (Twine(Partial) + Twine(Result)).str();

    // After this point we'll only be emitting parenthesis for the clockwise
    // spiral rule, or append square brackets at the end of Result for arrays.

    // Find the next non-array qualifier. Skip over const-qualifiers, because in
    // C there are no const-arrays, so we'll have to deal with const-arrays
    // separately.
    auto QPointerIt = std::find_if(QArrayIt, QEnd, model::Qualifier::isPointer);
    {
      // If we find the next pointer qualifier, go back to the first previous
      // const-qualifier that const-qualifies the pointer itself. This is
      // necessary, so that we can reason about the element of the array being
      // const, and we can deal properly with const arrays.
      if (QPointerIt != QEnd and QPointerIt != QArrayIt
          and isConst(*std::make_reverse_iterator(QPointerIt)))
        QPointerIt = std::prev(QPointerIt);
    }

    if (QArrayIt != QPointerIt) {
      // If QT is s a pointer to an array we have to add parentheses for the
      // clockwise spiral rule
      auto ReverseQArrayIt = std::make_reverse_iterator(QArrayIt);
      bool LastWasPointer = QArrayIt != QIt and isPointer(*ReverseQArrayIt);
      if (LastWasPointer)
        Result = (Twine("(") + Twine(Result) + Twine(")")).str();

      const auto &ArrayOrConstRange = llvm::make_range(QArrayIt, QPointerIt);
      bool ConstQualifiedArray = llvm::any_of(ArrayOrConstRange, isConst);

      // If the array is const-qualfied and its element is not const-qualified,
      // just print it as an array of const-qualified elements, because that's
      // the equivalent semantics in C anyway.
      if (ConstQualifiedArray) {
        bool ElementIsConstQualified = QPointerIt != QEnd
                                       and isConst(*QPointerIt);
        // If the array is const qualified but the element is not, we have to
        // force const-ness onto the element, because in C there's no way to
        // const-qualify arrays. If the element is already const-qualified, then
        // there's no need to do that, because we're still gonna print the
        // const-qualifier for the element.
        if (not ElementIsConstQualified) {

          const auto &Const = B.getKeyword(ptml::PTMLCBuilder::Keyword::Const)
                                .serialize();
          Result = (Twine(" ") + Twine(Const) + Twine(" ") + Twine(Result))
                     .str();
        }
      }

      for (const model::Qualifier &ArrayQ :
           llvm::reverse(llvm::make_filter_range(ArrayOrConstRange,
                                                 model::Qualifier::isArray)))
        Result.append((Twine("[") + Twine(ArrayQ.Size()) + Twine("]")).str());
    }

    QIt = QPointerIt;
  } while (QIt != QEnd);

  Result = (Twine(TypeName) + Twine(Result)).str();

  return Result;
}

TypeString getArrayWrapper(const model::QualifiedType &QT,
                           const ptml::PTMLCBuilder &B) {
  revng_assert(QT.isArray());
  TypeString Result;
  Result.append(ArrayWrapperPrefix);

  for (const auto &Qualifier : QT.Qualifiers()) {

    switch (Qualifier.Kind()) {

    case model::QualifierKind::Const: {
      Result.append("const_");
    } break;

    case model::QualifierKind::Pointer: {
      Result.append("ptr_to_");
    } break;

    case model::QualifierKind::Array: {
      auto NElem = Qualifier.Size();
      Result.append(("array_" + Twine(NElem) + "_of_").str());
    } break;

    default:
      revng_abort();
    }
  }

  Result.append(QT.UnqualifiedType().get()->name());
  Tag ResultTag = B.getTag(ptml::tags::Span, Result.str());
  return TypeString(ResultTag.serialize());
}

TypeString getNamedInstanceOfReturnType(const model::TypeDefinition &Function,
                                        llvm::StringRef InstanceName,
                                        const ptml::PTMLCBuilder &B,
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
               .serialize();
    if (not InstanceName.empty())
      Result.append((Twine(" ") + Twine(InstanceName)).str());
    break;

  case ReturnMethod::ModelAggregate:
  case ReturnMethod::Scalar: {
    model::QualifiedType ReturnType;

    if (ReturnMethod == ReturnMethod::ModelAggregate) {
      ReturnType = Layout.returnValueAggregateType();
    } else {
      revng_assert(Layout.ReturnValues.size() == 1);
      ReturnType = Layout.ReturnValues[0].Type;
    }

    // When returning arrays, they need to be wrapped into an artificial
    // struct
    if (ReturnType.isArray()) {
      Result = getArrayWrapper(ReturnType, B);
      if (not InstanceName.empty())
        Result.append((Twine(" ") + Twine(InstanceName)).str());
    } else {
      Result = getNamedCInstance(ReturnType, InstanceName, B, AllowedActions);
    }

  } break;

  case ReturnMethod::RegisterSet: {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    revng_assert(llvm::isa<model::RawFunctionDefinition>(Function));
    std::string Name = (Twine(RetStructPrefix) + Function.name()).str();
    std::string Location = pipeline::serializedLocation(ranks::ArtificialStruct,
                                                        Function.key());
    Result = B.tokenTag(Name, ptml::c::tokens::Type)
               .addAttribute(B.getLocationAttribute(IsDefinition), Location)
               .serialize();
    if (not InstanceName.empty())
      Result.append((Twine(" ") + Twine(InstanceName)).str());
  } break;

  default:
    revng_abort();
  }

  revng_assert(not llvm::StringRef(Result).trim().empty());
  return TypeString(B.getTag(ptml::tags::Span, Result)
                      .addAttribute(attributes::ActionContextLocation,
                                    serializedLocation(ranks::ReturnValue,
                                                       Function.key()))
                      .serialize());
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
                                       ptml::PTMLCBuilder &B,
                                       const model::Binary &Model,
                                       bool SingleLine) {
  using namespace abi::FunctionType;
  auto Layout = Layout::make(RF);
  revng_assert(not Layout.hasSPTAR());
  revng_assert(Layout.returnMethod() != ReturnMethod::ModelAggregate);

  Header << B.getAnnotateABI(("raw_"
                              + model::Architecture::getName(RF.Architecture()))
                               .str());
  if (Function and not Function->Attributes().empty())
    Header << getFunctionAttributesString(Function->Attributes());
  Header << (SingleLine ? " " : "\n");
  Header << getNamedInstanceOfReturnType(RF, FunctionName, B, false);

  if (RF.Arguments().empty() and RF.StackArgumentsType().empty()) {
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
        MarkedType = getNamedCInstance(Arg.Type(), ArgString, B).str().str();
      std::string
        MarkedReg = B.getAnnotateReg(model::Register::getName(Arg.Location()));
      Tag ArgTag = B.getTag(ptml::tags::Span, MarkedType + " " + MarkedReg);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          serializedLocation(ranks::RawArgument,
                                             RF.key(),
                                             Arg.key()));

      Header << Separator << ArgTag.serialize();
      Separator = Comma;
    }

    if (not RF.StackArgumentsType().empty()) {
      // Add last argument representing a pointer to the stack arguments
      std::string StackArgName;
      if (Function != nullptr)
        StackArgName = getArgumentLocationDefinition("_stack_arguments",
                                                     *Function,
                                                     B);
      Header << Separator
             << getNamedCInstance({ RF.StackArgumentsType(), {} },
                                  StackArgName,
                                  B);
      Header << " " << B.getAnnotateStack();
    }
    Header << ")";
  }
}

template<ModelFunction FunctionType>
static void printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::CABIFunctionDefinition &CF,
                                       const llvm::StringRef &FunctionName,
                                       llvm::raw_ostream &Header,
                                       ptml::PTMLCBuilder &B,
                                       const model::Binary &Model,
                                       bool SingleLine) {
  Header << B.getAnnotateABI(model::ABI::getName(CF.ABI()));
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
      if (Arg.Type().isArray()) {
        ArgDeclaration = getArrayWrapper(Arg.Type(), B);
        if (not ArgString.empty()) {
          ArgDeclaration.append(" ");
          ArgDeclaration.append(ArgString);
        }
      } else {
        ArgDeclaration = getNamedCInstance(Arg.Type(), ArgString, B);
      }

      Tag ArgTag = B.getTag(ptml::tags::Span, ArgDeclaration);
      ArgTag.addAttribute(attributes::ActionContextLocation,
                          serializedLocation(ranks::CABIArgument,
                                             CF.key(),
                                             Arg.key()));
      Header << Separator << ArgTag.serialize();
      Separator = Comma;
    }
    Header << ")";
  }
}

void printFunctionPrototype(const model::TypeDefinition &FT,
                            const model::Function &Function,
                            llvm::raw_ostream &Header,
                            ptml::PTMLCBuilder &B,
                            const model::Binary &Model,
                            bool SingleLine) {
  std::string Location = serializedLocation(ranks::Function, Function.key());
  Tag FunctionTag = B.tokenTag(Function.name(), ptml::c::tokens::Function)
                      .addAttribute(attributes::ActionContextLocation, Location)
                      .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.serialize(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.serialize(),
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
                            ptml::PTMLCBuilder &B,
                            const model::Binary &Model,
                            bool SingleLine) {
  std::string Location = serializedLocation(ranks::DynamicFunction,
                                            Function.key());
  Tag FunctionTag = B.tokenTag(Function.name(), ptml::c::tokens::Function)
                      .addAttribute(attributes::ActionContextLocation, Location)
                      .addAttribute(attributes::LocationDefinition, Location);
  if (auto *RF = dyn_cast<model::RawFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.serialize(),
                               Header,
                               B,
                               Model,
                               SingleLine);
  } else if (auto *CF = dyn_cast<model::CABIFunctionDefinition>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.serialize(),
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
                                  ptml::PTMLCBuilder &B,
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
