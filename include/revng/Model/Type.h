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
#include "revng/Model/Identifier.h"
#include "revng/Model/Register.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: Type
doc: Base class of model types used for LLVM-style RTTI
type: struct
fields:
  - name: Kind
    type: model::TypeKind::Values
  - name: ID
    type: uint64_t
key:
  - Kind
  - ID
abstract: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Type.h"

/// Concept to identify all types that are derived from model::Type
template<typename T>
concept IsModelType = std::is_base_of_v<model::Type, T>;

class model::Type : public model::generated::Type {
public:
  static constexpr const auto AssociatedKind = TypeKind::Invalid;

public:
  // TODO: Constructors cannot be inherited, since the default one is
  //  manually implemented in order to generate a random ID
  Type();
  Type(TypeKind::Values TK);
  Type(TypeKind::Values Kind, uint64_t ID) : model::generated::Type(Kind, ID) {}

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return true; }

  Identifier name() const;

public:
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

namespace model {

using UpcastableType = UpcastablePointer<model::Type>;

template<size_t I = 0>
inline model::UpcastableType
makeTypeWithID(model::TypeKind::Values Kind, uint64_t ID) {
  using concrete_types = concrete_types_traits_t<model::Type>;
  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = std::tuple_element_t<I, concrete_types>;
    if (type::classof(typename type::Key(Kind, ID)))
      return UpcastableType(new type(type::AssociatedKind, ID));
    else
      return model::makeTypeWithID<I + 1>(Kind, ID);
  } else {
    return UpcastableType(nullptr);
  }
}

using TypePath = TupleTreeReference<model::Type, model::Binary>;

template<IsModelType T, typename... Args>
inline UpcastableType makeType(Args &&...A) {
  return UpcastableType::make<T>(std::forward<Args>(A)...);
}

} // end namespace model

#include "revng/Model/Generated/Late/Type.h"
