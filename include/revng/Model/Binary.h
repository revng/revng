#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"

#include "revng/ADT/Concepts.h"
#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Configuration.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Function.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Register.h"
#include "revng/Model/Segment.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/MetaAddressRangeSet.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreeDiff.h"

/* TUPLE-TREE-YAML
name: Binary
doc: Data structure representing the whole binary
type: struct
fields:
  - name: Architecture
    doc: Binary architecture
    type: Architecture
    optional: true
  - name: EntryPoint
    doc: Program entry point
    type: MetaAddress
    optional: true
  - name: DefaultABI
    doc: The default ABI of `RawFunctionDefinition`s within the binary
    type: ABI
    optional: true
  - name: DefaultPrototype
    doc: The default function prototype
    reference:
      pointeeType: TypeDefinition
      rootType: Binary
    optional: true
  - name: Configuration
    type: Configuration
    optional: true
  - name: Segments
    doc: List of segments in the original binary
    sequence:
      type: SortedVector
      elementType: Segment
    optional: true
  - name: ExtraCodeAddresses
    doc: Addresses containing code in order to help translation
    optional: true
    sequence:
      type: SortedVector
      elementType: MetaAddress
  - name: ImportedLibraries
    doc: List of imported libraries
    sequence:
      type: SortedVector
      elementType: string
    optional: true
  - name: ImportedDynamicFunctions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: DynamicFunction
    optional: true
  - name: Functions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: Function
    optional: true
  - name: TypeDefinitions
    doc: The type system
    sequence:
      type: SortedVector
      upcastable: true
      elementType: TypeDefinition
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Binary.h"

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

class model::Binary : public model::generated::Binary {
public:
  using generated::Binary::Binary;

public:
  model::DefinitionReference
  getDefinitionReference(const model::TypeDefinition::Key &Key) {
    return DefinitionReference::fromString(this,
                                           "/TypeDefinitions/"
                                             + getNameFromYAMLScalar(Key));
  }

  model::DefinitionReference
  getDefinitionReference(const model::TypeDefinition::Key &Key) const {
    return DefinitionReference::fromString(this,
                                           "/TypeDefinitions/"
                                             + getNameFromYAMLScalar(Key));
  }

  model::DefinitionReference
  getDefinitionReference(const model::TypeDefinition *T) {
    return getDefinitionReference(T->key());
  }

  model::DefinitionReference
  getDefinitionReference(const model::TypeDefinition *T) const {
    return getDefinitionReference(T->key());
  }

  /// Return the first available (non-primitive) type ID available
  uint64_t getAvailableTypeID() const;

  /// Record the new type into the model and assign an ID (unless it's a
  /// PrimitiveDefinition)
  model::DefinitionReference recordNewType(model::UpcastableTypeDefinition &&T);

  /// Uses `SortedVector::batch_insert()` to emplace all the elements from
  /// \ref NewTypes range into the `TypeDefinitions()` set.
  ///
  /// This inserts all the elements at the end of the underlying vector, and
  /// then triggers sorting, instead of conventional searching for the position
  /// of each element on its insertion.
  ///
  /// \note Unlike recordNewTypeDefinitions, this method does not assign type
  /// IDs.
  ///
  /// \note It takes advantage of `std::move_iterator` to ensure all
  ///       the elements are accessed strictly as r-values, so the original
  ///       container, \ref NewTypes range points to, is left in an unspecified
  ///       state after the invocation, as all of its elements are moved out of.
  ///
  /// \note Since a strict version of `batch-insert`'er is used, if this causes
  ///       multiple elements to have the same \ref TypeDefinition::Key,
  ///       an assert will be fired.
  ///
  /// \tparam Range constrained input range type.
  /// \param  NewTypes the input range.
  template<range_with_value_type<UpcastablePointer<TypeDefinition>> Range>
  void recordNewTypeDefinitions(Range &&NewTypes) {
    auto Inserter = TypeDefinitions().batch_insert();

    static_assert(std::is_rvalue_reference_v<decltype(NewTypes)>);
    auto Movable = as_rvalue(std::move(NewTypes));
    for (UpcastablePointer<TypeDefinition> &&NewType : Movable) {
      static_assert(std::is_rvalue_reference_v<decltype(NewType)>);
      revng_assert(NewType->ID() != 0);
      Inserter.emplace(std::move(NewType));
    }
  }

  template<derived_from<model::TypeDefinition> NewType,
           typename... ArgumentTypes>
  [[nodiscard]] std::pair<NewType &, model::DefinitionReference>
  makeTypeDefinition(ArgumentTypes &&...Arguments) {
    using UT = model::UpcastableTypeDefinition;
    UT Result = UT::make<NewType>(std::forward<ArgumentTypes>(Arguments)...);
    model::DefinitionReference ResultPath = recordNewType(std::move(Result));
    return { *llvm::cast<NewType>(ResultPath.get()), ResultPath };
  }

  model::DefinitionReference getPrimitiveType(PrimitiveKind::Values V,
                                              uint8_t ByteSize);

  model::DefinitionReference getPrimitiveType(PrimitiveKind::Values V,
                                              uint8_t ByteSize) const;

  bool verifyTypeDefinitions() const debug_function;
  bool verifyTypeDefinitions(bool Assert) const debug_function;
  bool verifyTypeDefinitions(VerifyHelper &VH) const;

public:
  std::string path(const model::Function &F) const {
    return "/Functions/" + key(F);
  }

  std::string path(const model::DynamicFunction &F) const {
    return "/ImportedDynamicFunctions/" + key(F);
  }

  std::string path(const model::TypeDefinition &T) const {
    return "/TypeDefinitions/" + key(T);
  }

  std::string path(const model::EnumDefinition &D,
                   const model::EnumEntry &Entry) const {
    return path(static_cast<const model::TypeDefinition &>(D)) + "/Entries/"
           + key(Entry);
  }

  std::string path(const model::Segment &Segment) const {
    return "/Segments/" + key(Segment);
  }

public:
  MetaAddressRangeSet executableRanges() const;

public:
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  bool verify() const;
  void dumpTypeGraph(const char *Path) const debug_function;
  std::string toString() const debug_function;

private:
  template<typename T>
  static std::string key(const T &Object) {
    return getNameFromYAMLScalar(KeyedObjectTraits<T>::key(Object));
  }
};

#include "revng/Model/Generated/Late/Binary.h"
