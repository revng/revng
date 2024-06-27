#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/CommonTypeMethods.h"
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

class model::Type : public model::generated::Type,
                    public model::CommonTypeMethods<Type> {
public:
  static constexpr const auto AssociatedKind = TypeKind::Invalid;

public:
  using generated::Type::Type;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;

public:
  /// Tries to recursively compute the size of the type.
  ///
  /// Use this method only on (temporarily) invalid models, for example within
  /// the binary importer. If your model is valid, use \ref size() instead.
  ///
  /// \returns * `std::nullopt` if the size cannot be computed, for example,
  ///            when the type system loops and the type's size depends on
  ///            the type itself,
  ///          * 0 for types without the size (`void` and function types),
  ///          * size in bytes in all other cases.
  ///
  /// @{
  std::optional<uint64_t> trySize() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> trySize(VerifyHelper &VH) const;
  /// @}

public:
  /// If the innermost type is defined, skip all the wrappers (arrays and
  /// pointers) and return its definition. Otherwise, return a nullptr.
  ///
  /// This is provided for some very specific cases, please take extra care when
  /// using it. An example of such a case is establishing of relation between
  /// types (for example to build a graph of dependencies).
  ///
  /// If you are hesitating whether you need this, please shop around for other
  /// helpers (like `classification helpers` group from the common type methods)
  /// If any of them sounds like what you want, prefer it instead.
  ///
  /// \returns a pointer to the definition, or a `nullptr` for primitive types
  ///          that do not have one.
  ///
  /// @{
  model::TypeDefinition *skipToDefinition();
  const model::TypeDefinition *skipToDefinition() const;
  /// @}

  /// If the innermost type is defined, skip all the wrappers (arrays and
  /// pointers) and return it. Otherwise, return a nullptr.
  ///
  /// This is provided for some very specific cases, please take extra care when
  /// using it. An example of such a case is replacing root pointers within
  /// the defined types.
  ///
  /// @{
  model::DefinedType *skipToDefinedType();
  const model::DefinedType *skipToDefinedType() const;
  /// @}

public:
  std::strong_ordering operator<=>(const model::Type &Another) const;
  bool operator==(const model::Type &Another) const {
    return (*this <=> Another) == std::strong_ordering::equal;
  }
};

#include "revng/Model/Generated/Late/Type.h"

namespace model {

/// Return a copy of this type with constness information stripped away.
/// If it's impossible to do so because of a typedef, it is unwrapped, as in
/// a copy of the typedef's underlying type without constness information is
/// returned instead.
model::UpcastableType getNonConst(const model::Type &);

template<typename T>
concept AnyType = std::derived_from<T, model::Type>
                  || std::derived_from<T, model::TypeDefinition>;

} // namespace model
