//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

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
namespace tokens = ptml::c::tokenTypes;
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
  TypeString Result;
  bool LastPointer = false;
  const model::Type *Unqualified = QT.UnqualifiedType.getConst();

  Result = getTypeName(*Unqualified);

  auto QIt = QT.Qualifiers.rbegin();
  auto QEnd = QT.Qualifiers.rend();
  for (; QIt != QEnd and not model::Qualifier::isArray(*QIt); ++QIt) {
    switch (QIt->Kind) {
    case model::QualifierKind::Const:
      LastPointer = false;
      Result.append({ " ",
                      Tag(tags::Span, "const")
                        .addAttribute(attributes::Token, tokens::Operator)
                        .serialize() });
      break;
    case model::QualifierKind::Pointer:
      Result.append({ LastPointer ? "" : " ",
                      Tag(tags::Span, "*")
                        .addAttribute(attributes::Token, tokens::Operator)
                        .serialize() });
      LastPointer = true;
      break;
    default:
      revng_abort();
    }
  }

  if (!LastPointer && !InstanceName.empty())
    Result.append(" ");

  Result.append(InstanceName.str());

  for (; QIt != QEnd; ++QIt) {
    // TODO: We would actually want to assert:
    //   revng_assert(model::Qualifier::isArray(*QIt));
    // but at the moment we can't, because e.g. debug info imported from DWARF
    // allow specifying both a const array and an array of const.
    // The first is not emittable in C, the second is.
    // Because DWARF allows specifying this, we can end up with const qualifiers
    // in positions where C does not allow us to emit them.
    // In principle we could assert hard, but we would need a pre-processing
    // stage that massages the model so that all array types can be emitted in
    // C. At the moment this is a workaround that drops const qualifiers on
    // arrays.
    revng_assert(not model::Qualifier::isPointer(*QIt));
    Result.append((Twine("[") + Twine(QIt->Size) + Twine("]")).str());
  }

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
    Result = tokenTag((Twine(RetStructPrefix) + "returned_by_"
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
    Header << "(" << tokenTag("void", tokens::Type) << ")";
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
      model::QualifiedType StackArgsPtr = RF.StackArgumentsType;
      addPointerQualifier(StackArgsPtr, Model);
      Header << Separator << getNamedCInstance(StackArgsPtr, StackVarsName);
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
    Header << "(" << tokenTag("void", tokens::Type) << ")";
  } else {
    const StringRef Open = "(";
    const StringRef Comma = ", ";
    StringRef Separator = Open;

    for (const auto &Arg : CF.Arguments) {
      TypeString ArgTypeName;
      if (Arg.Type.isArray())
        ArgTypeName = getArrayWrapper(Arg.Type);
      else
        ArgTypeName = getNamedCInstance(Arg.Type, "");

      std::string ArgumentName = ArgumentPrinter(Arg);
      if (!ArgumentName.empty())
        Header << Separator << ArgTypeName << " " << ArgumentName;
      else
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
  Tag FunctionTag = tokenTag(Function.name(), tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(Function))
                      .addAttribute(Declaration ?
                                      attributes::LocationDefinition :
                                      attributes::LocationReferences,
                                    serializedLocation(ranks::Function,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const NamedTypedRegister &Reg) {
      return tokenTag(Reg.name().str(), tokens::FunctionParameter)
        .addAttribute(attributes::LocationDefinition,
                      serializedLocation(ranks::RawFunctionArgument,
                                         Function.key(),
                                         Reg.key()))
        .serialize();
    };

    std::string
      StackName = tokenTag("stack_args", tokens::FunctionParameter)
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
      return tokenTag(Arg.name().str(), tokens::FunctionParameter)
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
  Tag FunctionTag = tokenTag(Function.name(), tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(Function))
                      .addAttribute(Declaration ?
                                      attributes::LocationDefinition :
                                      attributes::LocationReferences,
                                    serializedLocation(ranks::DynamicFunction,
                                                       Function.key()));
  if (auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    auto ArgumentPrinter = [&](const NamedTypedRegister &Reg) {
      return tokenTag(Reg.name().str(), tokens::FunctionParameter)
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
      return tokenTag(Arg.name().str(), tokens::FunctionParameter)
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
