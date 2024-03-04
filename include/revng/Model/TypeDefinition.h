#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/ABI.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: TypeDefinition
doc: Base class of model type definitions used for LLVM-style RTTI
type: struct
fields:
  - name: ID
    type: uint64_t
    is_guid: true
  - name: Kind
    type: TypeDefinitionKind
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
key:
  - ID
  - Kind
abstract: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypeDefinition.h"

class model::TypeDefinition : public model::generated::TypeDefinition,
                              public model::CommonTypeMethods<TypeDefinition> {
public:
  using generated::TypeDefinition::TypeDefinition;

  Identifier name() const;

  const llvm::SmallVector<const model::Type *, 4> edges() const;
  void dumpTypeGraph(const char *Path) const debug_function;

public:
  bool verify(bool Assert = false) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

namespace model {
using DefinitionReference = TupleTreeReference<model::TypeDefinition,
                                               model::Binary>;
}

extern template model::DefinitionReference
model::DefinitionReference::fromString<model::Binary>(model::Binary *Root,
                                                      llvm::StringRef Path);

extern template model::DefinitionReference
model::DefinitionReference::fromString<const model::Binary>(const model::Binary
                                                              *,
                                                            llvm::StringRef);

#include "revng/Model/Generated/Late/TypeDefinition.h"
