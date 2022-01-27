#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"

/* TUPLE-TREE-YAML
name: TypedefType
doc: A typedef type in model
type: struct
inherits: Type
tag: Typedef
fields:
  - name: UnderlyingType
    type: model::QualifiedType
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypedefType.h"

class model::TypedefType : public model::generated::TypedefType {
public:
  static constexpr const char *AutomaticNamePrefix = "typedef_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Typedef;

public:
  using generated::TypedefType::TypedefType;
  TypedefType() : generated::TypedefType() { Kind = AssociatedKind; }

public:
  Identifier name() const;

public:
  llvm::SmallVector<model::QualifiedType, 4> edges() {
    return { UnderlyingType };
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/TypedefType.h"
