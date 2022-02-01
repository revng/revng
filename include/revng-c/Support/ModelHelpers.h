#pragma once

#include "llvm/ADT/STLExtras.h"

#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

/// Strip off all the possible layers of constness and typedefs from QT
extern model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT, model::VerifyHelper &VH);

/// Strip off all the possible layers of constness and typedefs from QT
inline model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT) {
  model::VerifyHelper VH;
  return peelConstAndTypedefs(QT, VH);
}

/// Create an empty model::StructType of size Size in Binary
extern model::TypePath createEmptyStruct(model::Binary &Binary, uint64_t Size);

/// Convert an LLVM integer type (i1, i8, i16, ...) to the corresponding
/// primitive type (uint8_t, uint16_t, ...).
extern const model::QualifiedType
llvmIntToModelType(const llvm::Type *LLVMType, const model::Binary &Model);

/// Parse a QualifiedType from a string.
extern model::QualifiedType
parseQualifiedType(const llvm::StringRef QTString, const model::Binary &Model);

/// Add a pointer Qualifier of the right dimension to a given \a QT
inline void
addPointerQualifier(model::QualifiedType &QT, const model::Binary &Binary) {
  using Qualifier = model::Qualifier;
  QT.Qualifiers.push_back(Qualifier::createPointer(Binary.Architecture));
}

/// Create a pointer to the given base Type
inline model::QualifiedType
createPointerTo(const model::TypePath &BaseT, const model::Binary &Binary) {
  using Qualifier = model::Qualifier;

  return model::QualifiedType{
    BaseT, { Qualifier::createPointer(Binary.Architecture) }
  };
}

/// Return a new type if \a QT is a pointer, otherwise return \a QT
extern model::QualifiedType dropPointer(const model::QualifiedType &QT);

/// Drops te last pointer qualifier from \a QT or, if this wraps a
/// typedef, it recursively descends the typedef wrappers until a pointer
/// qualifier is found.
extern RecursiveCoroutine<model::QualifiedType>
dropPointerRecursively(const model::QualifiedType &QT);
