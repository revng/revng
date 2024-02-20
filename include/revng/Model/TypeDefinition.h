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
#include "revng/Model/QualifiedType.h"
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

class model::TypeDefinition : public model::generated::TypeDefinition {
public:
  static constexpr const auto AssociatedKind = TypeDefinitionKind::Invalid;

public:
  using generated::TypeDefinition::TypeDefinition;

  TypeDefinition();
  TypeDefinition(uint64_t ID, TypeDefinitionKind::Values Kind);

public:
  static bool classof(const TypeDefinition *D) { return classof(D->key()); }
  static bool classof(const Key &K) { return true; }

  Identifier name() const;

public:
  /// Recursively computes size of the type.
  ///
  /// It asserts in cases where the size cannot be computed, for example, when
  /// the type system loops and the type's size depends on the type itself.
  ///
  /// \returns * `std::nullopt` if the type does not have size (for example,
  ///            it's a `void` primitive or a function type),
  ///          * size in bytes otherwise.
  std::optional<uint64_t> size() const debug_function;
  std::optional<uint64_t> size(VerifyHelper &VH) const;

  /// Tries to recursively compute the size of the type.
  ///
  /// Use this method only on (temporarily) invalid models. You most likely want
  /// to use size().
  ///
  /// \returns * `std::nullopt` if the size cannot be computed, for example,
  ///            when the type system loops and the type's size depends on
  ///            the type itself,
  ///          * 0 for types without the size (`void` and function types),
  ///          * size in bytes in all other cases.
  std::optional<uint64_t> trySize() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> trySize(VerifyHelper &VH) const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
  void dumpTypeGraph(const char *Path) const debug_function;
};

namespace model {

using UpcastableTypeDefinition = UpcastablePointer<model::TypeDefinition>;

template<std::derived_from<model::TypeDefinition> T, typename... Args>
inline model::UpcastableTypeDefinition makeTypeDefinition(Args &&...A) {
  return model::UpcastableTypeDefinition::make<T>(std::forward<Args>(A)...);
}

} // end namespace model

extern template model::TypeDefinitionPath
model::TypeDefinitionPath::fromString<model::Binary>(model::Binary *Root,
                                                     llvm::StringRef Path);

extern template model::TypeDefinitionPath
model::TypeDefinitionPath::fromString<const model::Binary>(const model::Binary
                                                             *,
                                                           llvm::StringRef);

#include "revng/Model/Generated/Late/TypeDefinition.h"
