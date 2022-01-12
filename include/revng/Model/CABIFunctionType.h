#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"
#include "revng/Model/Argument.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"

/* TUPLE-TREE-YAML
name: CABIFunctionType
doc: |
  The function type described through a C-like prototype plus an ABI.

  This is an "high level" representation of the prototype of a function. It is
  expressed as list of arguments composed by an index and a type. No
  information about the register is embedded. That information is implicit in
  the ABI this type is associated to.
type: struct
inherits: Type
fields:
  - name: CustomName
    type: Identifier
    optional: true
  - name: ABI
    type: model::ABI::Values
  - name: ReturnType
    type: model::QualifiedType
  - name: Arguments
    sequence:
      type: SortedVector
      elementType: model::Argument
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/CABIFunctionType.h"

class model::CABIFunctionType : public model::generated::CABIFunctionType {
public:
  static constexpr const char *AutomaticNamePrefix = "cabifunction_";
  static constexpr const TypeKind::Values
    AssociatedKind = TypeKind::CABIFunctionType;

public:
  using generated::CABIFunctionType::CABIFunctionType;
  CABIFunctionType() : generated::CABIFunctionType() { Kind = AssociatedKind; }

public:
  Identifier name() const;
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/CABIFunctionType.h"
