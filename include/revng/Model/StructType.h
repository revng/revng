#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/StructField.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: StructType
doc: |
  A struct type in model.
  Structs are actually typedefs of unnamed structs in C.
type: struct
inherits: Type
fields:
  - name: Size
    doc: Size in bytes
    type: uint64_t
  - name: Fields
    sequence:
      type: SortedVector
      elementType: StructField
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructType.h"

class model::StructType : public model::generated::StructType {
public:
  static constexpr const char *AutomaticNamePrefix = "struct_";

public:
  using generated::StructType::StructType;
  StructType() : generated::StructType() {}

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (auto &Field : Fields())
      Result.push_back(Field.Type());

    return Result;
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/StructType.h"
