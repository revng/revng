#pragma once

#include "revng/Model/Binary.h"
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
