//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <optional>
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
#include "revng/Support/CDataModel.h"

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

  model::UpcastableType makeStandardPrimitive(const BuiltinType *Builtin);

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

/// The C standard type a builtin is spelled as, if the data model gives that
/// spelling a size. The signed and unsigned members of a pair share an entry:
/// only the width is read from here, the signedness comes from the builtin.
static std::optional<CStandardType>
getStandardType(const BuiltinType *Builtin) {
  switch (Builtin->getKind()) {
  case BuiltinType::Char_S:
  case BuiltinType::Char_U:
  case BuiltinType::SChar:
  case BuiltinType::UChar:
    return CStandardType::Char;

  case BuiltinType::Short:
  case BuiltinType::UShort:
    return CStandardType::Short;

  case BuiltinType::Int:
  case BuiltinType::UInt:
    return CStandardType::Int;

  case BuiltinType::Long:
  case BuiltinType::ULong:
    return CStandardType::Long;

  case BuiltinType::LongLong:
  case BuiltinType::ULongLong:
    return CStandardType::LongLong;

  case BuiltinType::Float:
    return CStandardType::Float;

  case BuiltinType::Double:
    return CStandardType::Double;

  case BuiltinType::LongDouble:
    return CStandardType::LongDouble;

  default:
    return std::nullopt;
  }
}

/// Converts a builtin spelled as a plain C type, such as `unsigned long`, into
/// a primitive whose size is the one the target ABI gives that type.
model::UpcastableType
Converter::makeStandardPrimitive(const BuiltinType *Builtin) {
  if (Builtin->getKind() == BuiltinType::Void)
    return model::PrimitiveType::makeVoid();

  std::optional<CStandardType> Standard = getStandardType(Builtin);
  if (not Standard) {
    PrintingPolicy Policy(Context.getLangOpts());
    Errors.emplace_back(ErrorPrefix.str() + " Builtin type `"
                        + Builtin->getName(Policy).str()
                        + "` is not supported, please use a revng "
                          "model::PrimitiveType instead.\n");

    return model::UpcastableType::empty();
  }

  uint64_t Size = Binary.targetDataModel().getStandardTypeSize(*Standard);
  if (isFloatingPointType(*Standard))
    return model::PrimitiveType::makeFloat(Size);

  revng_assert(isIntegerType(*Standard));

  // A plain `char` is whichever the parse says it is: `Char_S` under
  // `-fsigned-char`, which is how we parse C, and `Char_U` otherwise.
  return Builtin->isUnsignedInteger() ?
           model::PrimitiveType::makeUnsigned(Size) :
           model::PrimitiveType::makeSigned(Size);
}

model::UpcastableType
Converter::makePrimitive(const BuiltinType *UnderlyingBuiltin, QualType Type) {
  revng_assert(UnderlyingBuiltin);

  auto AsElaboratedType = Type->getAs<ElaboratedType>();
  if (not AsElaboratedType)
    return makeStandardPrimitive(UnderlyingBuiltin);

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
