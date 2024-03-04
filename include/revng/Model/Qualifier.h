#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"
#include "revng/Model/QualifierKind.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: Qualifier
doc: A qualifier for a model::TypeDefinition
type: struct
fields:
  - name: Kind
    type: QualifierKind
  - name: Size
    doc: Size in bytes for Pointer, number of elements for Array, 0 otherwise
    type: uint64_t
    optional: true
key:
  - Kind
  - Size
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Qualifier.h"

class model::Qualifier : public model::generated::Qualifier {
public:
  using generated::Qualifier::Qualifier;

public:
  // Kind is not Invalid, Pointer and Const have no Size, Array has Size.
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;

public:
  static Qualifier createConst() { return Qualifier(QualifierKind::Const, 0); }

  static Qualifier createPointer(uint64_t Size) {
    return Qualifier(QualifierKind::Pointer, Size);
  }

  static Qualifier createPointer(model::Architecture::Values Architecture) {
    return createPointer(getPointerSize(Architecture));
  }

  static Qualifier createArray(uint64_t S) {
    return Qualifier(QualifierKind::Array, S);
  }

public:
  static bool isConst(const Qualifier &Q) {
    revng_assert(Q.verify(true));
    return Q.Kind() == QualifierKind::Const;
  }

  static bool isArray(const Qualifier &Q) {
    revng_assert(Q.verify(true));
    return Q.Kind() == QualifierKind::Array;
  }

  static bool isPointer(const Qualifier &Q) {
    revng_assert(Q.verify(true));
    return Q.Kind() == QualifierKind::Pointer;
  }

public:
  auto operator<(const Qualifier &Other) const {
    auto Me = std::tie(this->Kind(), this->Size());
    auto It = std::tie(Other.Kind(), Other.Size());
    return Me < It;
  }
};

#include "revng/Model/Generated/Late/Qualifier.h"
