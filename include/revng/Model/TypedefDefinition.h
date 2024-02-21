#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: TypedefDefinition
doc: A typedef type definition in model
type: struct
inherits: TypeDefinition
fields:
  - name: UnderlyingType
    type: QualifiedType
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypedefDefinition.h"

class model::TypedefDefinition : public model::generated::TypedefDefinition {
public:
  static constexpr const char *AutomaticNamePrefix = "typedef_";

public:
  using generated::TypedefDefinition::TypedefDefinition;

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    return { UnderlyingType() };
  }
};

#include "revng/Model/Generated/Late/TypedefDefinition.h"
