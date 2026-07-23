//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ClangToModel/QualTypeToModel.h"
#include "revng/Model/ArrayType.h"
#include "revng/Model/Binary.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/TypeDefinitionByName.h"
#include "revng/Support/Assert.h"

using namespace llvm;
using namespace clang;

static constexpr llvm::StringRef PrimitiveTypeHeader = "primitive-types.h";

namespace {

/// Holds the state a single conversion needs: the model to resolve named
/// definitions against, the Clang context, an error sink, and the prefix each
/// diagnostic is tagged with (so the calling analysis owns the wording).
struct Converter {
  const model::Binary &Binary;
  clang::ASTContext &Context;
  std::vector<std::string> &Errors;
  llvm::StringRef ErrorPrefix;

  model::UpcastableType makePrimitive(const BuiltinType *UnderlyingBuiltin,
                                      QualType Type);

  template<NonBaseDerived<model::TypeDefinition> T>
  model::UpcastableType makeTypeByNameOrID(llvm::StringRef Name);

  model::UpcastableType
  getTypeForRecordType(const clang::RecordType *RecordType,
                       const QualType &ClangType);

  model::UpcastableType getTypeForEnumType(const clang::EnumType *EnumType);

  bool comesFromPrimitiveTypesHeader(const clang::RecordDecl *RD);

  RecursiveCoroutine<model::UpcastableType> convert(const QualType &QT);
};

model::UpcastableType
Converter::makePrimitive(const BuiltinType *UnderlyingBuiltin, QualType Type) {
  revng_assert(UnderlyingBuiltin);

  auto AsElaboratedType = Type->getAs<ElaboratedType>();
  if (not AsElaboratedType) {
    PrintingPolicy Policy(Context.getLangOpts());
    Errors.emplace_back(ErrorPrefix.str() + " Builtin type `"
                        + UnderlyingBuiltin->getName(Policy).str()
                        + "` not allowed, please use a revng "
                          "model::PrimitiveType instead.\n");

    return model::UpcastableType::empty();
  }

  while (auto Typedef = AsElaboratedType->getAs<TypedefType>()) {
    auto TheUnderlyingType = Typedef->getDecl()->getUnderlyingType();
    if (not TheUnderlyingType->getAs<ElaboratedType>())
      break;
    AsElaboratedType = TheUnderlyingType->getAs<ElaboratedType>();
  }

  std::string TypeName = AsElaboratedType->getNamedType().getAsString();
  if (model::PrimitiveType::fromCName(TypeName).isEmpty()) {
    Errors.emplace_back(ErrorPrefix.str() + " `"
                        + AsElaboratedType->getNamedType().getAsString()
                        + "` type is not supported, please use a revng "
                          "model::PrimitiveType instead.\n");

    return model::UpcastableType::empty();
  }

  switch (UnderlyingBuiltin->getKind()) {
  case BuiltinType::UInt128:
    return model::PrimitiveType::makeUnsigned(16);

  case BuiltinType::Int128:
    return model::PrimitiveType::makeSigned(16);

  case BuiltinType::ULongLong:
  case BuiltinType::ULong:
    return model::PrimitiveType::makeUnsigned(8);

  case BuiltinType::LongLong:
  case BuiltinType::Long:
    return model::PrimitiveType::makeSigned(8);

  case BuiltinType::WChar_U:
  case BuiltinType::UInt:
    return model::PrimitiveType::makeUnsigned(4);

  case BuiltinType::WChar_S:
  case BuiltinType::Char32:
  case BuiltinType::Int:
    return model::PrimitiveType::makeSigned(4);

  case BuiltinType::UShort:
    return model::PrimitiveType::makeUnsigned(2);

  case BuiltinType::Char16:
  case BuiltinType::Short:
    return model::PrimitiveType::makeSigned(2);

  case BuiltinType::Char_U:
  case BuiltinType::UChar:
  case BuiltinType::Char8:
  case BuiltinType::Bool:
    return model::PrimitiveType::makeUnsigned(1);

  case BuiltinType::Char_S:
  case BuiltinType::SChar:
    return model::PrimitiveType::makeSigned(1);

  case BuiltinType::Void:
    return model::PrimitiveType::makeVoid();

  case BuiltinType::Float16:
    return model::PrimitiveType::makeFloat(2);

  case BuiltinType::Float:
    return model::PrimitiveType::makeFloat(4);

  case BuiltinType::Double:
    return model::PrimitiveType::makeFloat(8);

  case BuiltinType::Float128:
  case BuiltinType::LongDouble:
    return model::PrimitiveType::makeFloat(16);

  default:
    Errors.emplace_back(ErrorPrefix.str()
                        + " Unable to handle a primitive type.\n");
  }

  return model::UpcastableType::empty();
}

template<NonBaseDerived<model::TypeDefinition> T>
model::UpcastableType Converter::makeTypeByNameOrID(llvm::StringRef Name) {
  return model::getTypeDefinitionByNameOrID(Binary, Name, T::AssociatedKind);
}

model::UpcastableType
Converter::getTypeForRecordType(const clang::RecordType *RecordType,
                                const QualType &ClangType) {
  revng_assert(RecordType);

  // Check if it is a primitive type described with a struct.
  if (comesFromPrimitiveTypesHeader(RecordType->getDecl())) {
    const TypedefType *AsTypedef = ClangType->getAs<TypedefType>();
    if (not AsTypedef) {
      Errors.emplace_back(ErrorPrefix.str()
                          + " There should be a typedef for struct that "
                            "defines the primitive type.\n");
      return model::UpcastableType::empty();
    }
    auto TypeName = AsTypedef->getDecl()->getName();
    auto R = model::PrimitiveType::fromCName(TypeName);
    revng_assert(R);
    return R;
  }

  auto Name = RecordType->getDecl()->getName();
  if (Name.empty()) {
    Errors.emplace_back(ErrorPrefix.str()
                        + " Nameless structs and unions are not supported "
                          "here, since we have no way to trace them back to "
                          "one "
                          "of the types present in the model.\n");
    return model::UpcastableType::empty();
  }

  if (RecordType->isStructureType()) {
    if (auto Struct = makeTypeByNameOrID<model::StructDefinition>(Name))
      return Struct;

  } else if (RecordType->isUnionType()) {
    if (auto Union = makeTypeByNameOrID<model::UnionDefinition>(Name))
      return Union;
  }

  Errors.emplace_back(ErrorPrefix.str() + " Unknown struct or union: `"
                      + Name.str() + "`.\n");
  return model::UpcastableType::empty();
}

model::UpcastableType
Converter::getTypeForEnumType(const clang::EnumType *EnumType) {
  revng_assert(EnumType);

  auto EnumName = EnumType->getDecl()->getName();
  if (EnumName.empty()) {
    Errors.emplace_back(ErrorPrefix.str()
                        + " Nameless enums are not supported here, since we "
                          "have no way to trace them back to one of the types "
                          "present in the model.\n");
    return model::UpcastableType::empty();
  }

  if (auto Enum = makeTypeByNameOrID<model::EnumDefinition>(EnumName))
    return Enum;

  Errors.emplace_back(ErrorPrefix.str() + " Unknown enum: `" + EnumName.str()
                      + "`.\n");
  return model::UpcastableType::empty();
}

bool Converter::comesFromPrimitiveTypesHeader(const clang::RecordDecl *RD) {
  SourceManager &SM = Context.getSourceManager();
  PresumedLoc Loc = SM.getPresumedLoc(RD->getLocation());
  if (!Loc.isValid())
    return false;

  StringRef TheFileName(Loc.getFilename());
  if (TheFileName.contains(PrimitiveTypeHeader))
    return true;

  return false;
}

RecursiveCoroutine<model::UpcastableType>
Converter::convert(const QualType &QT) {
  model::UpcastableType R;

  if (const BuiltinType *AsBuiltinType = QT->getAs<BuiltinType>()) {
    R = makePrimitive(AsBuiltinType, QT);

  } else if (const PointerType *Pointer = QT->getAs<PointerType>()) {
    QualType Pointee = Pointer->getPointeeType();
    R = model::PointerType::make(rc_recur convert(Pointee),
                                 Binary.Architecture());

  } else if (QT->isArrayType()) {
    // getAsConstantArrayType, unlike dyn_cast, looks through sugar such as the
    // parentheses in a pointer-to-array type.
    if (const auto *CAT = Context.getAsConstantArrayType(QT)) {
      QualType ElementType = CAT->getElementType();
      uint64_t NumberOfElements = CAT->getSize().getZExtValue();
      R = model::ArrayType::make(rc_recur convert(ElementType),
                                 NumberOfElements);
    } else {
      // Here we can face `clang::VariableArrayType` and
      // `clang::IncompleteArrayType`.
      Errors.emplace_back(ErrorPrefix.str()
                          + " Unsupported type used as an array.\n");
    }
  } else if (const RecordType *AsRecordType = QT->getAs<RecordType>()) {
    R = getTypeForRecordType(AsRecordType, QT);

  } else if (const EnumType *AsEnum = QT->getAs<EnumType>()) {
    R = getTypeForEnumType(AsEnum);

  } else if (const auto *AsFn = QT->getAs<FunctionProtoType>()) {
    if (const TypedefType *AsTypedef = QT->getAs<TypedefType>()) {
      auto Name = AsTypedef->getDecl()->getName();
      if (auto CFT = makeTypeByNameOrID<model::CABIFunctionDefinition>(Name))
        R = std::move(CFT);
      else if (auto Rw = makeTypeByNameOrID<model::RawFunctionDefinition>(Name))
        R = std::move(Rw);
      else
        Errors.emplace_back(ErrorPrefix.str() + " Unknown typedef: `"
                            + Name.str() + "`.\n");
    } else {
      Errors.emplace_back(ErrorPrefix.str()
                          + " Model has to contain a typedef for the function "
                            "prototype.\n");
    }

  } else {
    Errors.emplace_back(ErrorPrefix.str()
                        + " The type cannot be represented in the model.\n");
  }

  if (not R.isEmpty() and QT.isConstQualified())
    R->IsConst() = true;

  rc_return R;
}

} // namespace

RecursiveCoroutine<model::UpcastableType>
revng::qualTypeToModel(const clang::QualType &QT,
                       const model::Binary &Binary,
                       clang::ASTContext &Context,
                       std::vector<std::string> &Errors,
                       llvm::StringRef ErrorPrefix) {
  Converter TheConverter{ Binary, Context, Errors, ErrorPrefix };
  rc_return rc_recur TheConverter.convert(QT);
}
