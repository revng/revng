#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: QualifiedType
doc: A qualified version of a model::Type. Can have many nested qualifiers
type: struct
fields:
  - name: UnqualifiedType
    reference:
      pointeeType: model::Type
      rootType: model::Binary
  - name: Qualifiers
    sequence:
      type: "std::vector"
      elementType: model::Qualifier
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/QualifiedType.h"

class model::QualifiedType : public model::generated::QualifiedType {
public:
  using generated::QualifiedType::QualifiedType;

public:
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

  bool isScalar() const;
  bool isPrimitive(model::PrimitiveTypeKind::Values V) const;
  bool isVoid() const { return isPrimitive(model::PrimitiveTypeKind::Void); }
  bool isFloat() const { return isPrimitive(model::PrimitiveTypeKind::Float); }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/QualifiedType.h"
