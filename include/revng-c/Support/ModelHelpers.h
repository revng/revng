#pragma once

#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

/// \brief Strip off all the possible layers of constness and typedefs from QT
extern model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT, model::VerifyHelper &VH);

/// \brief Strip off all the possible layers of constness and typedefs from QT
inline model::QualifiedType
peelConstAndTypedefs(const model::QualifiedType &QT) {
  model::VerifyHelper VH;
  return peelConstAndTypedefs(QT, VH);
}

/// \brief Create an empty model::StructType of size Size in Binary
extern model::TypePath createEmptyStruct(model::Binary &Binary, uint64_t Size);

/// \brief Check if the given type has to be serialized in C as an array.
extern bool isEventuallyArray(const model::QualifiedType &QT);
