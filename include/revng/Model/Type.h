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
#include "revng/Model/TypeKind.h"
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
    type: TypeKind
  - name: ID
    type: uint64_t
    is_guid: true
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
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
  Type(TypeKind::Values Kind, uint64_t ID) : Type(Kind, ID, Identifier(), "") {}
  Type(TypeKind::Values Kind,
       uint64_t ID,
       Identifier CustomName,
       std::string OriginalName) :
    model::generated::Type(Kind, ID, CustomName, OriginalName) {}

public:
  static bool classof(const Type *T) { return classof(T->key()); }
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
  /// \returns * `std::nullopt` if the size cannot be computed, for example,
  ///            when the type system loops and the type's size depends on
  ///            the type itself,
  ///          * 0 for types without the size (e.g. `void`),
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

using UpcastableType = UpcastablePointer<model::Type>;

model::UpcastableType makeTypeWithID(model::TypeKind::Values Kind, uint64_t ID);

using TypePath = TupleTreeReference<model::Type, model::Binary>;

template<IsModelType T, typename... Args>
inline UpcastableType makeType(Args &&...A) {
  return UpcastableType::make<T>(std::forward<Args>(A)...);
}

} // end namespace model

extern template model::TypePath
model::TypePath::fromString<model::Binary>(model::Binary *Root,
                                           llvm::StringRef Path);

extern template model::TypePath
model::TypePath::fromString<const model::Binary>(const model::Binary *Root,
                                                 llvm::StringRef Path);

#include "revng/Model/Generated/Late/Type.h"
