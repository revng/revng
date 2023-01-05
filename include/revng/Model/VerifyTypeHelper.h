#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/QualifiedType.h"
#include "revng/Model/TypedefType.h"

inline bool isOnlyConstQualified(const model::QualifiedType &QT) {
  if (QT.Qualifiers().empty() or QT.Qualifiers().size() > 1)
    return false;

  return model::Qualifier::isConst(QT.Qualifiers()[0]);
}

struct VoidConstResult {
  bool IsVoid;
  bool IsConst;
};

inline VoidConstResult isVoidConst(const model::QualifiedType *QualType) {
  using namespace model;

  revng_assert(QualType != nullptr);

  VoidConstResult Result{ false, false };

  while (true) {
    // If the argument type is qualified try to get the unqualified version.
    // Warning: we only skip const-qualifiers here, cause the other qualifiers
    // actually produce a different type.
    const Type *UnqualType = nullptr;
    if (not QualType->Qualifiers().empty()) {

      // If it has a non-const qualifier, it can never be void because it's a
      // pointer or array, so we can break out.
      if (not isOnlyConstQualified(*QualType)) {
        return Result;
      }

      // We know that it's const-qualified here, and it only has one
      // qualifier, hence we can skip the const-qualifier.
      Result.IsConst = true;
      return Result;
    }

    if (QualType->isPrimitive2()) {
      Result.IsVoid = QualType->PrimitiveKind() == PrimitiveTypeKind::Void;
      return Result;
    } else {
      revng_assert(not QualType->UnqualifiedType().empty());

      UnqualType = QualType->UnqualifiedType().get();

      if (UnqualType->Kind() == TypeKind::TypedefType) {
        // If we still have a typedef in our way, unwrap it and keep looking.
        QualType = &llvm::cast<TypedefType>(UnqualType)->UnderlyingType();
      } else {
        // In all the other cases it's not void, break from the while.
        return Result;
      }
    }

  }

  revng_abort();
}
