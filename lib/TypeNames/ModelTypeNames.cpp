//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/ModelHelpers.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;
using llvm::Twine;
using tokenDefinition::types::TypeString;

using modelEditPath::getCustomNamePath;
using pipeline::serializedLocation;
using ptml::str;
using ptml::Tag;
namespace tags = ptml::tags;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace ranks = revng::ranks;

using namespace ArtificialTypes;

template<typename FT>
// clang-format off
concept ModelFunction = std::same_as<FT, model::Function>
                        or std::same_as<FT, model::DynamicFunction>;
// clang-format on

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
static std::string
getArgumentLocation(llvm::StringRef ArgumentName, const FunctionType &F) {
  return Tag(tags::Span, ArgumentName)
    .addAttribute(attributes::Token, tokens::FunctionParameter)
    .addAttribute(ptml::locationAttribute(IsDefinition),
                  serializeVariableLocation(ArgumentName, F))
    .serialize();
}

static std::string
getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                              const model::DynamicFunction &F) {
  return getArgumentLocation<true>(ArgumentName, F);
}

static std::string getArgumentLocationDefinition(llvm::StringRef ArgumentName,
                                                 const model::Function &F) {
  return getArgumentLocation<true>(ArgumentName, F);
}

std::string getArgumentLocationReference(llvm::StringRef ArgumentName,
                                         const model::Function &F) {
  return getArgumentLocation<false>(ArgumentName, F);
}

template<bool IsDefinition>
static std::string
getVariableLocation(llvm::StringRef VariableName, const model::Function &F) {
  return Tag(tags::Span, VariableName)
    .addAttribute(attributes::Token, tokens::Variable)
    .addAttribute(ptml::locationAttribute(IsDefinition),
                  serializeVariableLocation(VariableName, F))
    .serialize();
}

std::string getVariableLocationDefinition(llvm::StringRef VariableName,
                                          const model::Function &F) {
  return getVariableLocation<true>(VariableName, F);
}

std::string getVariableLocationReference(llvm::StringRef VariableName,
                                         const model::Function &F) {
  return getVariableLocation<false>(VariableName, F);
}

TypeString getReturnField(const model::RawFunctionType &F, size_t Index) {
  revng_assert(F.ReturnValues.size() > 1);
  return TypeString((Twine(RetFieldPrefix) + Twine(Index)).str());
}

TypeString
getNamedCInstance(const model::QualifiedType &QT, StringRef InstanceName) {

  const auto &isConst = model::Qualifier::isConst;
  const auto &isPointer = model::Qualifier::isPointer;

  TypeString Result;

  // Here we have a bunch of pointers, const, and array qualifiers.
  // Because of arrays, we have to emit the types with C infamous clockwise
  // spiral rule. Luckily all our function types have names, so at least this
  // cannot become too nasty.

  auto QIt = QT.Qualifiers.begin();
  auto QEnd = QT.Qualifiers.end();
  do {
    // Find the first qualifer that is an array.
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
          Result.append(" ");

        switch (Q.Kind) {

        case model::QualifierKind::Const:
          Result.append(keywords::Const.serialize());
          PrevPointer = false;
          break;
        case model::QualifierKind::Pointer:
          Result.append(Tag(tags::Span, "*")
                          .addAttribute(attributes::Token, tokens::Operator)
                          .serialize());
          PrevPointer = true;
          break;

        default:
          revng_abort();
        }
      }
    }

    bool IsPointer = QArrayIt != QIt
                     and isPointer(*std::make_reverse_iterator(QArrayIt));

    // Print the actual instance name.
    if (QIt == QT.Qualifiers.begin() and not InstanceName.empty()) {
      if (not IsPointer)
        Result.append(" ");
      Result.append(InstanceName.str());
    }

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
      if (IsPointer)
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

          const auto &Const = keywords::Const.serialize();
          Result = (Twine(" ") + Twine(Const) + Twine(" ") + Twine(Result))
                     .str();
        }
      }

      for (const model::Qualifier &ArrayQ :
           llvm::reverse(llvm::make_filter_range(ArrayOrConstRange,
                                                 model::Qualifier::isArray)))
        Result.append((Twine("[") + Twine(ArrayQ.Size) + Twine("]")).str());
    }

    QIt = QPointerIt;
  } while (QIt != QEnd);

  const model::Type &Unqualified = *QT.UnqualifiedType.getConst();
  std::string TypeName = ptml::getLocationReference(Unqualified);
  Result = (Twine(TypeName) + Twine(" ") + Twine(Result)).str();

  return Result;
}

TypeString getArrayWrapper(const model::QualifiedType &QT) {
  revng_assert(QT.isArray());
  TypeString Result;
  Result.append(ArrayWrapperPrefix);

  for (const auto &Qualifier : QT.Qualifiers) {

    switch (Qualifier.Kind) {

    case model::QualifierKind::Const: {
      Result.append("const_");
    } break;

    case model::QualifierKind::Pointer: {
      Result.append("ptr_to_");
    } break;

    case model::QualifierKind::Array: {
      auto NElem = Qualifier.Size;
      Result.append(("array_" + Twine(NElem) + "_of_").str());
    } break;

    default:
      revng_abort();
    }
  }

  Result.append(QT.UnqualifiedType.get()->name());
  Tag ResultTag = Tag(tags::Span, Result.str());
  return TypeString(ResultTag.serialize());
}

TypeString getReturnTypeName(const model::RawFunctionType &F) {
  TypeString Result;

  if (F.ReturnValues.size() == 0) {
    Result = Tag(tags::Span, "void")
               .addAttribute(attributes::Token, tokens::Type)
               .serialize();
  } else if (F.ReturnValues.size() == 1) {
    auto RetTy = F.ReturnValues.begin()->Type;
    // RawFunctionTypes should never be returning an array
    revng_assert(not RetTy.isArray());
    Result = getNamedCInstance(RetTy, "");
  } else {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    Result = ptml::tokenTag((Twine(RetStructPrefix) + "returned_by_"
                             + model::Identifier::fromString(F.name()))
                              .str(),
                            tokens::Type)
               .serialize();
  }

  revng_assert(not Result.empty());
  return Result;
}

TypeString getReturnTypeName(const model::CABIFunctionType &F) {
  TypeString Result;
  const auto &RetTy = F.ReturnType;

  if (RetTy.isArray()) {
    // Returned arrays get wrapped in an artificial struct
    Result = getArrayWrapper(RetTy);
  } else {
    Result = getNamedCInstance(RetTy, "");
  }

  revng_assert(not Result.empty());
  return Result;
}

template<ModelFunction FunctionType>
static void printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::RawFunctionType &RF,
                                       const llvm::StringRef &FunctionName,
                                       llvm::raw_ostream &Header,
                                       const model::Binary &Model,
                                       bool Declaration) {
  Header << getReturnTypeName(RF);
  if (RF.ReturnValues.size() == 1) {
    const model::QualifiedType &ReturnType = RF.ReturnValues.begin()->Type;
    if (not ReturnType.isPointer() or ReturnType.isConst())
      Header << " ";
  } else {
    Header << " ";
  }
  Header << FunctionName;

  revng_assert(RF.StackArgumentsType.Qualifiers.empty());
  if (RF.Arguments.empty()
      and not RF.StackArgumentsType.UnqualifiedType.isValid()) {
    Header << "(" << ptml::tokenTag("void", tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;
    for (const auto &Arg : RF.Arguments) {
      auto ArgName = model::Identifier::fromString(Arg.name()).str().str();
      std::string ArgString = Function ?
                                getArgumentLocationDefinition(ArgName,
                                                              *Function) :
                                "";
      Header << Separator << getNamedCInstance(Arg.Type, ArgString);
      Separator = Comma;
    }

    revng_assert(RF.StackArgumentsType.Qualifiers.empty());
    if (RF.StackArgumentsType.UnqualifiedType.isValid()) {
      // Add last argument representing a pointer to the stack arguments
      auto StackArgName = Function ? getArgumentLocationDefinition("stack_args",
                                                                   *Function) :
                                     "";
      Header << Separator
             << getNamedCInstance(Model.getPointerTo(RF.StackArgumentsType),
                                  StackArgName);
    }
    Header << ")";
  }
}

template<ModelFunction FunctionType>
static void printFunctionPrototypeImpl(const FunctionType *Function,
                                       const model::CABIFunctionType &CF,
                                       const llvm::StringRef &FunctionName,
                                       llvm::raw_ostream &Header,
                                       const model::Binary &Model,
                                       bool Declaration) {
  Header << getReturnTypeName(CF);
  if (not CF.ReturnType.isPointer() or CF.ReturnType.isConst())
    Header << " ";
  Header << FunctionName;

  if (CF.Arguments.empty()) {
    Header << "(" << ptml::tokenTag("void", tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Arg : CF.Arguments) {
      auto ArgName = model::Identifier::fromString(Arg.name()).str().str();
      std::string ArgString = Function ?
                                getArgumentLocationDefinition(ArgName,
                                                              *Function) :
                                "";
      TypeString ArgDeclaration;
      if (Arg.Type.isArray()) {
        ArgDeclaration = getArrayWrapper(Arg.Type);
        if (not ArgString.empty()) {
          ArgDeclaration.append(" ");
          ArgDeclaration.append(ArgString);
        }
      } else {
        ArgDeclaration = getNamedCInstance(Arg.Type, ArgString);
      }
      Header << Separator << ArgDeclaration;
      Separator = Comma;
    }
    Header << ")";
  }
}

void printFunctionPrototype(const model::Type &FT,
                            const model::Function &Function,
                            llvm::raw_ostream &Header,
                            const model::Binary &Model,
                            bool Declaration) {

  Tag FunctionTag = ptml::tokenTag(Function.name(), tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(Function))
                      .addAttribute(ptml::locationAttribute(Declaration),
                                    serializedLocation(ranks::Function,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {

    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.serialize(),
                               Header,
                               Model,
                               Declaration);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {

    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.serialize(),
                               Header,
                               Model,
                               Declaration);
  } else {
    revng_abort();
  }
}

void printFunctionPrototype(const model::Type &FT,
                            const model::DynamicFunction &Function,
                            llvm::raw_ostream &Header,
                            const model::Binary &Model,
                            bool Declaration) {
  Tag FunctionTag = ptml::tokenTag(Function.name(), tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(Function))
                      .addAttribute(ptml::locationAttribute(Declaration),
                                    serializedLocation(ranks::DynamicFunction,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *RF,
                               FunctionTag.serialize(),
                               Header,
                               Model,
                               Declaration);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    printFunctionPrototypeImpl(&Function,
                               *CF,
                               FunctionTag.serialize(),
                               Header,
                               Model,
                               Declaration);
  } else {
    revng_abort();
  }
}

void printFunctionTypeDeclaration(const model::Type &FT,
                                  llvm::raw_ostream &Header,
                                  const model::Binary &Model) {

  auto TypeName = ptml::getLocationDefinition(FT);
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    printFunctionPrototypeImpl<model::Function>(nullptr,
                                                *RF,
                                                TypeName,
                                                Header,
                                                Model,
                                                true);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    printFunctionPrototypeImpl<model::Function>(nullptr,
                                                *CF,
                                                TypeName,
                                                Header,
                                                Model,
                                                true);
  } else {
    revng_abort();
  }
}
