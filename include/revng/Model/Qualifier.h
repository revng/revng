#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/QualifierKind.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: Qualifier
doc: A qualifier for a model::Type
type: struct
fields:
  - name: Kind
    type: QualifierKind::Values
  - name: Size
    doc: Size in bytes for Pointer, number of elements for Array, 0 otherwise
    type: uint64_t
    optional: true
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
  void dump() const debug_function;

public:
  static Qualifier createConst() { return Qualifier(QualifierKind::Const, 0); }

  static Qualifier createPointer(uint64_t Size) {
    return Qualifier(QualifierKind::Pointer, Size);
  }

  static Qualifier createArray(uint64_t S) {
    return Qualifier(QualifierKind::Array, S);
  }

public:
  bool isConstQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Const;
  }

  bool isArrayQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Array;
  }

  bool isPointerQualifier() const {
    revng_assert(verify(true));
    return Kind == QualifierKind::Pointer;
  }

public:
  auto operator<(const Qualifier &Other) const {
    return this->Kind < Other.Kind && this->Size < Other.Size;
  }
};

#include "revng/Model/Generated/Late/Qualifier.h"
