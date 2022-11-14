#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/QualifiedType.h"

namespace {

bool isOnlyConstQualified(const model::QualifiedType &QT) {
  if (QT.Qualifiers().empty() or QT.Qualifiers().size() > 1)
    return false;

  return model::Qualifier::isConst(QT.Qualifiers()[0]);
}

struct VoidConstResult {
  bool IsVoid;
  bool IsConst;
};

VoidConstResult isVoidConst(const model::QualifiedType *QualType) {
  VoidConstResult Result{ /* IsVoid */ false, /* IsConst */ false };

  bool Done = false;
  while (not Done) {

    // If the argument type is qualified try to get the unqualified version.
    // Warning: we only skip const-qualifiers here, cause the other qualifiers
    // actually produce a different type.
    const model::Type *UnqualType = nullptr;
    if (not QualType->Qualifiers().empty()) {

      // If it has a non-const qualifier, it can never be void because it's a
      // pointer or array, so we can break out.
      if (not isOnlyConstQualified(*QualType)) {
        Done = true;
        continue;
      }

      // We know that it's const-qualified here, and it only has one
      // qualifier, hence we can skip the const-qualifier.
      Result.IsConst = true;
      return Result;
    }
    UnqualType = QualType->UnqualifiedType().get();

    switch (UnqualType->Kind()) {

    // If we still have a typedef in our way, unwrap it and keep looking.
    case model::TypeKind::TypedefType: {
      QualType = &llvm::cast<model::TypedefType>(UnqualType)->UnderlyingType();
    } break;

    // If we have a primitive type, check the name, and we're done.
    case model::TypeKind::PrimitiveType: {
      auto *P = llvm::cast<model::PrimitiveType>(UnqualType);
      Result.IsVoid = P->PrimitiveKind() == model::PrimitiveTypeKind::Void;
      Done = true;
    } break;

    // In all the other cases it's not void, break from the while.
    default: {
      Done = true;
    } break;
    }
  }
  return Result;
}
} // namespace
