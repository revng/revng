#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/PrimitiveKind.h"

namespace model {
class VerifyHelper;

class TypeDefinition;
class StructDefinition;

class Type;
class ArrayType;
class PointerType;
class PrimitiveType;

template<typename CRTP>
class CommonTypeMethods {
  // TODO: this class could _really_ take advantage of deducing `this`.
  //       Please, adopt it as soon as it's available.

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
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

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
  /// This is a strict version of \ref skipToDefinition. This does not ignore
  /// any qualifiers, asserting if one is met.
  ///
  /// Only use this in places where the type is guaranteed (as in, verified) to
  /// be unqualified.
  ///
  /// \note you can use \ref isDefinition() to check if this would assert or,
  ///       preferably, \ref tryGetAsDefinition(), which returns a `nullptr`, if
  ///       this would have asserted.
  model::TypeDefinition &asDefinition() { return *tryGetAsDefinition(); }
  const auto &asDefinition() const { return *tryGetAsDefinition(); }
  model::TypeDefinition *tryGetAsDefinition();
  const model::TypeDefinition *tryGetAsDefinition() const;
  bool isDefinition() const { return tryGetAsDefinition() != nullptr; }

  /// This is a specialized version of \ref asDefinition(), see
  /// the documentation there for more details.
  model::TypeDefinition &asPrototype() { return *tryGetAsPrototype(); }
  const auto &asPrototype() const { return *tryGetAsPrototype(); }
  model::TypeDefinition *tryGetAsPrototype();
  const model::TypeDefinition *tryGetAsPrototype() const;
  bool isPrototype() const { return tryGetAsPrototype() != nullptr; }

  /// This is a specialized version of \ref asDefinition(), see
  /// the documentation there for more details.
  model::StructDefinition &asStruct() { return *tryGetAsStruct(); }
  const model::StructDefinition &asStruct() const { return *tryGetAsStruct(); }
  model::StructDefinition *tryGetAsStruct();
  const model::StructDefinition *tryGetAsStruct() const;
  bool isStruct() const { return tryGetAsStruct() != nullptr; }

  /// This is a specialized version of \ref asDefinition(), see
  /// the documentation there for more details.
  model::PrimitiveType &asPrimitive() { return *tryGetAsPrimitive(); }
  const auto &asPrimitive() const { return *tryGetAsPrimitive(); }
  model::PrimitiveType *tryGetAsPrimitive();
  const model::PrimitiveType *tryGetAsPrimitive() const;
  bool isPrimitive() const { return tryGetAsPrimitive() != nullptr; }

  /// This is a specialized version of \ref asDefinition(), see
  /// the documentation there for more details.
  model::ArrayType &asArray() { return *tryGetAsArray(); }
  const model::ArrayType &asArray() const { return *tryGetAsArray(); }
  model::ArrayType *tryGetAsArray();
  const model::ArrayType *tryGetAsArray() const;
  bool isArray() const { return tryGetAsArray() != nullptr; }

  /// This is a specialized version of \ref asDefinition(), see
  /// the documentation there for more details.
  model::PointerType &asPointer() { return *tryGetAsPointer(); }
  const model::PointerType &asPointer() const { return *tryGetAsPointer(); }
  model::PointerType *tryGetAsPointer();
  const model::PointerType *tryGetAsPointer() const;
  bool isPointer() const { return tryGetAsPointer() != nullptr; }

  bool isScalar() const;

public:
  /// Use this method to check whether a specific type is constant or not, as
  /// opposed to the \ref IsConst() member (which should be reserved for
  /// assigning `const`ness), since it unwraps typedefs and arrays.
  ///
  /// \note If you ever need a way to get rid of constness, please introduce
  ///       a dedicated `dropConst` helper here as opposed to doing that at
  ///       the call site.
  bool isConst() const;

  /// Skip all the "wrappers" around the definition.
  ///
  /// \returns a pointer to the definition, or a `nullptr` for primitive types
  ///          that do not have one.
  ///
  /// \note in the cases when this returns `nullptr`, it's safe to just call
  ///       `asPrimitive`.
  model::TypeDefinition *skipToDefinition();
  const model::TypeDefinition *skipToDefinition() const;

public:
  /// Returns the pointee type while unwrapping typedefs.
  ///
  /// \note This asserts if the type is _not_ a pointer, so only use it when
  ///       it's guaranteed to be one.
  ///
  /// \note This also ignores the constness of the dropped pointer (and typedefs
  ///       in the way)
  model::Type &stripPointer();
  const model::Type &stripPointer() const;

private:
  using PKind = model::PrimitiveKind::Values;

public:
  bool isPrimitive(model::PrimitiveKind::Values Kind) const;
  bool isVoid() const { return isPrimitive(PKind::Void); }
  bool isGeneric() const { return isPrimitive(PKind::Generic); }
  bool isPointerOrNumber() const { return isPrimitive(PKind::PointerOrNumber); }
  bool isNumber() const { return isPrimitive(PKind::Number); }
  bool isUnsigned() const { return isPrimitive(PKind::Unsigned); }
  bool isSigned() const { return isPrimitive(PKind::Signed); }
  bool isFloat() const { return isPrimitive(PKind::Float); }

public:
  /// The helper for typedef unwrapping.
  ///
  /// \returns - nullptr if called with a non-typedef definition, since it's not
  ///            possible to produce a `model::Type` for one of those without
  ///            placing it into a binary.
  ///          - a pointer to an unwrapped `model::Type` in all the other cases.
  model::Type *skipTypedefs();
  const model::Type *skipTypedefs() const;

  /// The helper for unwrapping typedefs, including const ones.
  /// See \ref skipTypedef documentation for specifics.
  model::Type *skipConstAndTypedefs();
  const model::Type *skipConstAndTypedefs() const;

private:
  /// The CRTP unwrapping helper.
  CRTP &get() { return static_cast<CRTP &>(*this); }
  const CRTP &get() const { return static_cast<const CRTP &>(*this); }
};

} // namespace model
