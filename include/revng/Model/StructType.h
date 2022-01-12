#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/StructField.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"

/* TUPLE-TREE-YAML
name: StructType
doc: |
  A struct type in model.
  Structs are actually typedefs of unnamed structs in C.
type: struct
inherits: Type
tag: Struct
fields:
  - name: CustomName
    type: Identifier
    optional: true
  - name: Fields
    sequence:
      type: SortedVector
      elementType: model::StructField
  - name: Size
    doc: Size in bytes
    type: uint64_t
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructType.h"

class model::StructType : public model::generated::StructType {
public:
  static constexpr const char *AutomaticNamePrefix = "struct_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Struct;

public:
  using generated::StructType::StructType;
  StructType() : generated::StructType() { Kind = AssociatedKind; }

public:
  Identifier name() const;
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/StructType.h"
