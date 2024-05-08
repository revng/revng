#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: TypedefType
doc: A typedef type in model
type: struct
inherits: Type
fields:
  - name: UnderlyingType
    type: QualifiedType
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypedefType.h"

class model::TypedefType : public model::generated::TypedefType {
public:
  static constexpr const char *AutomaticNamePrefix = "typedef_";

public:
  using generated::TypedefType::TypedefType;
  TypedefType() : generated::TypedefType() {}

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    return { UnderlyingType() };
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/TypedefType.h"
