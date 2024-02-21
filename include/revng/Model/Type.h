#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: Type
doc: Base class of model types used for LLVM-style RTTI
type: struct
fields:
  - name: Kind
    type: TypeKind
  - name: IsConst
    type: bool
    optional: true
abstract: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Type.h"

class model::Type : public model::generated::Type {
public:
  static constexpr const auto AssociatedKind = TypeKind::Invalid;

public:
  using generated::Type::Type;

public:
  bool verify(bool Assert = false) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/Type.h"

namespace model {

template<typename T>
concept AnyType = std::derived_from<T, model::Type>
                  || std::derived_from<T, model::TypeDefinition>;

} // namespace model
