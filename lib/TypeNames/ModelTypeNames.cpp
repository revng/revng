//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"

#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using llvm::dyn_cast;
using llvm::isa;
using llvm::StringRef;
using llvm::Twine;

using namespace ArtificialTypes;

TypeString getReturnField(const model::RawFunctionType &F, size_t Index) {
  revng_assert(F.ReturnValues.size() > 1);
  return TypeString((Twine(RetFieldPrefix) + Twine(Index)).str());
}

TypeString getTypeName(const model::Type &T) {
  TypeString Result;

  if (isa<model::RawFunctionType>(&T) or isa<model::CABIFunctionType>(&T)) {
    Result.append(ArtificialTypes::FunctionTypedefPrefix);
    Result.append(model::Identifier::fromString(T.name()));

  } else {
    Result.append(T.name());
  }
  return Result;
}

TypeString
getNamedCInstance(const model::QualifiedType &QT, StringRef InstanceName) {
  TypeString Result;
  const model::Type *Unqualified = QT.UnqualifiedType.getConst();

  Result = getTypeName(*Unqualified);

  auto QIt = QT.Qualifiers.rbegin();
  auto QEnd = QT.Qualifiers.rend();
  bool PointerFound = false;
  for (; QIt != QEnd and not model::Qualifier::isArray(*QIt); ++QIt) {
    switch (QIt->Kind) {
    case model::QualifierKind::Const:
      Result.append(" const");
      break;
    case model::QualifierKind::Pointer:
      Result.append(" *");
      PointerFound = true;
      break;
    default:
      revng_abort();
    }
  }

  if (not Result.empty() and Result.back() != '*')
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
  return Result;
}

TypeString getReturnTypeName(const model::RawFunctionType &F) {
  TypeString Result;

  if (F.ReturnValues.size() == 0) {
    Result = "void";
  } else if (F.ReturnValues.size() == 1) {
    auto RetTy = F.ReturnValues.begin()->Type;
    // RawFunctionTypes should never be returning an array
    revng_assert(not RetTy.isArray());
    Result = getNamedCInstance(RetTy, "");
  } else {
    // RawFunctionTypes can return multiple values, which need to be wrapped
    // in a struct
    Result = (Twine(RetStructPrefix) + "returned_by_"
              + model::Identifier::fromString(F.name()))
               .str();
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

void printFunctionPrototype(const model::Type &FT,
                            StringRef FunctionName,
                            llvm::raw_ostream &Header) {

  if (const auto *RF = dyn_cast<model::RawFunctionType>(&FT)) {
    Header << getReturnTypeName(*RF) << " " << FunctionName.str();

    if (RF->Arguments.empty()) {
      Header << "(void)";
    } else {
      const StringRef Open = "(";
      const StringRef Comma = ", ";
      StringRef Separator = Open;
      for (const auto &Arg : RF->Arguments) {
        Header << Separator << getNamedCInstance(Arg.Type, Arg.name());
        Separator = Comma;
      }
      Header << ")";
    }
  } else if (const auto *CF = dyn_cast<model::CABIFunctionType>(&FT)) {
    Header << getReturnTypeName(*CF) << FunctionName;

    if (CF->Arguments.empty()) {
      Header << "(void)";

    } else {
      const StringRef Open = "(";
      const StringRef Comma = ", ";
      StringRef Separator = Open;

      for (const auto &Arg : CF->Arguments) {
        TypeString ArgTypeName;
        if (Arg.Type.isArray())
          ArgTypeName = getArrayWrapper(Arg.Type);
        else
          ArgTypeName = getNamedCInstance(Arg.Type, "");

        Header << Separator << ArgTypeName << " " << Arg.name();
        Separator = Comma;
      }
      Header << ")";
    }
  } else {
    revng_abort();
  }
}
