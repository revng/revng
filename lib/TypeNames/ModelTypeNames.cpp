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

TypeString getReturnField(const model::RawFunctionType &F, size_t Index) {
  revng_assert(F.ReturnValues.size() > 1);
  return TypeString((Twine(RetFieldPrefix) + Twine(Index)).str());
}

TypeString getTypeName(const model::Type &T) {
  Tag Result;

  if (isa<model::RawFunctionType>(&T) or isa<model::CABIFunctionType>(&T)) {
    TypeString Name;
    Name.append(ArtificialTypes::FunctionTypedefPrefix);
    Name.append(model::Identifier::fromString(T.name()));
    Result = Tag(tags::Span, Name.str())
               .addAttribute(attributes::ModelEditPath, getCustomNamePath(T));
  } else if (isa<model::PrimitiveType>(&T)) {
    Result = Tag(tags::Span, T.name().str());
  } else {
    Result = Tag(tags::Span, T.name().str())
               .addAttribute(attributes::ModelEditPath, getCustomNamePath(T));
  }
  Result.addAttribute(attributes::Token, tokens::Type)
    .addAttribute(attributes::LocationReferences,
                  serializedLocation(ranks::Type, T.key()));
  return TypeString(Result.serialize());
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
          Result.append(Tag(tags::Span, "const")
                          .addAttribute(attributes::Token, tokens::Operator)
                          .serialize());
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
          const auto &Const = Tag(tags::Span, "const")
                                .addAttribute(attributes::Token,
                                              tokens::Operator)
                                .serialize();
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

  TypeString UnqualifiedName = getTypeName(*QT.UnqualifiedType.getConst());
  Result = (Twine(UnqualifiedName) + Twine(" ") + Twine(Result)).str();

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

using model::NamedTypedRegister;
using std::function;
using RawArgumentPrinter = function<std::string(const NamedTypedRegister &)>;
static void printFunctionPrototypeImpl(const model::RawFunctionType &RF,
                                       const llvm::StringRef &FunctionName,
                                       RawArgumentPrinter ArgumentPrinter,
                                       const llvm::StringRef StackVarsName,
                                       llvm::raw_ostream &Header,
                                       const model::Binary &Model,
                                       bool Declaration) {
  Header << getReturnTypeName(RF) << " " << FunctionName;

  revng_assert(RF.StackArgumentsType.Qualifiers.empty());
  if (RF.Arguments.empty()
      and not RF.StackArgumentsType.UnqualifiedType.isValid()) {
    Header << "(" << ptml::tokenTag("void", tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;
    for (const auto &Arg : RF.Arguments) {
      std::string ArgumentName = ArgumentPrinter(Arg);
      Header << Separator << getNamedCInstance(Arg.Type, ArgumentName);
      Separator = Comma;
    }

    revng_assert(RF.StackArgumentsType.Qualifiers.empty());
    if (RF.StackArgumentsType.UnqualifiedType.isValid()) {
      // Add last argument representing a pointer to the stack arguments
      Header << Separator
             << getNamedCInstance(Model.getPointerTo(RF.StackArgumentsType),
                                  StackVarsName);
    }
    Header << ")";
  }
}

using model::Argument;
using CABIArgumentPrinter = std::function<std::string(const Argument &)>;
static void printFunctionPrototypeImpl(const model::CABIFunctionType &CF,
                                       const llvm::StringRef &FunctionName,
                                       CABIArgumentPrinter ArgumentPrinter,
                                       llvm::raw_ostream &Header,
                                       const model::Binary &Model,
                                       bool Declaration) {
  Header << getReturnTypeName(CF) << " " << FunctionName;

  if (CF.Arguments.empty()) {
    Header << "(" << ptml::tokenTag("void", tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Arg : CF.Arguments) {
      TypeString ArgTypeName;
      std::string ArgumentName = ArgumentPrinter(Arg);
      if (Arg.Type.isArray()) {
        ArgTypeName = getArrayWrapper(Arg.Type);
        if (not ArgumentName.empty()) {
          ArgTypeName.append(" ");
          ArgTypeName.append(ArgumentName);
        }
      } else {
        ArgTypeName = getNamedCInstance(Arg.Type, ArgumentName);
      }
      Header << Separator << ArgTypeName;
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
                      .addAttribute(Declaration ?
                                      attributes::LocationDefinition :
                                      attributes::LocationReferences,
                                    serializedLocation(ranks::Function,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const NamedTypedRegister &Reg) {
      return ptml::tokenTag(Reg.name().str(), tokens::FunctionParameter)
        .addAttribute(attributes::LocationDefinition,
                      serializedLocation(ranks::RawFunctionArgument,
                                         Function.key(),
                                         Reg.key()))
        .serialize();
    };

    std::string
      StackName = ptml::tokenTag("stack_args", tokens::FunctionParameter)
                    .addAttribute(attributes::LocationDefinition,
                                  serializedLocation(ranks::SpecialVariable,
                                                     Function.key(),
                                                     "stack_args"))
                    .serialize();

    printFunctionPrototypeImpl(*RF,
                               FunctionTag.serialize(),
                               ArgumentPrinter,
                               StackName,
                               Header,
                               Model,
                               Declaration);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const Argument &Arg) {
      return ptml::tokenTag(Arg.name().str(), tokens::FunctionParameter)
        .addAttribute(attributes::LocationDefinition,
                      serializedLocation(ranks::CABIFunctionArgument,
                                         Function.key(),
                                         Arg.key()))
        .serialize();
    };

    printFunctionPrototypeImpl(*CF,
                               FunctionTag.serialize(),
                               ArgumentPrinter,
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
                      .addAttribute(Declaration ?
                                      attributes::LocationDefinition :
                                      attributes::LocationReferences,
                                    serializedLocation(ranks::DynamicFunction,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const NamedTypedRegister &Reg) {
      return ptml::tokenTag(Reg.name().str(), tokens::FunctionParameter)
        .addAttribute(attributes::LocationDefinition,
                      serializedLocation(ranks::RawDynFunctionArgument,
                                         Function.key(),
                                         Reg.key()))
        .serialize();
    };

    printFunctionPrototypeImpl(*RF,
                               FunctionTag.serialize(),
                               ArgumentPrinter,
                               "stack_args",
                               Header,
                               Model,
                               Declaration);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const Argument &Arg) {
      return ptml::tokenTag(Arg.name().str(), tokens::FunctionParameter)
        .addAttribute(attributes::LocationDefinition,
                      serializedLocation(ranks::CABIDynFunctionArgument,
                                         Function.key(),
                                         Arg.key()))
        .serialize();
    };

    printFunctionPrototypeImpl(*CF,
                               FunctionTag.serialize(),
                               ArgumentPrinter,
                               Header,
                               Model,
                               Declaration);
  } else {
    revng_abort();
  }
}

void printFunctionPrototype(const model::Type &FT,
                            const llvm::StringRef &FunctionName,
                            llvm::raw_ostream &Header,
                            const model::Binary &Model,
                            bool Declaration) {
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const NamedTypedRegister &Reg) { return ""; };
    printFunctionPrototypeImpl(*RF,
                               FunctionName,
                               ArgumentPrinter,
                               "stack_args",
                               Header,
                               Model,
                               Declaration);
  } else if (auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const Argument &Arg) { return ""; };
    printFunctionPrototypeImpl(*CF,
                               FunctionName,
                               ArgumentPrinter,
                               Header,
                               Model,
                               Declaration);
  } else {
    revng_abort();
  }
}
