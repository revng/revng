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
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/Function.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Register.h"
#include "revng/Model/Segment.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreeDiff.h"

/* TUPLE-TREE-YAML
name: Binary
doc: Data structure representing the whole binary
type: struct
fields:
  - name: Functions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: Function
    optional: true
  - name: ImportedDynamicFunctions
    doc: List of the functions within the binary
    sequence:
      type: SortedVector
      elementType: DynamicFunction
    optional: true
  - name: Architecture
    doc: Binary architecture
    type: Architecture
    optional: true
  - name: DefaultABI
    doc: The default ABI of `RawFunctionType`s within the binary
    type: ABI
    optional: true
  - name: DefaultPrototype
    doc: The default function prototype
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
  - name: Segments
    doc: List of segments in the original binary
    sequence:
      type: SortedVector
      elementType: Segment
    optional: true
  - name: EntryPoint
    doc: Program entry point
    type: MetaAddress
    optional: true
  - name: Types
    doc: The type system
    sequence:
      type: SortedVector
      upcastable: true
      elementType: Type
    optional: true
  - name: ImportedLibraries
    doc: List of imported libraries
    sequence:
      type: SortedVector
      elementType: string
    optional: true
  - name: ExtraCodeAddresses
    doc: Addresses containing code in order to help translation
    optional: true
    sequence:
      type: SortedVector
      elementType: MetaAddress
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Binary.h"

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

namespace model {
using TypePath = TupleTreeReference<model::Type, model::Binary>;
}

class model::Binary : public model::generated::Binary {
public:
  using generated::Binary::Binary;

public:
  model::TypePath getTypePath(const model::Type::Key &Key) {
    return TypePath::fromString(this, "/Types/" + getNameFromYAMLScalar(Key));
  }

  model::TypePath getTypePath(const model::Type::Key &Key) const {
    return TypePath::fromString(this, "/Types/" + getNameFromYAMLScalar(Key));
  }

  model::TypePath getTypePath(const model::Type *T) {
    return getTypePath(T->key());
  }

  model::TypePath getTypePath(const model::Type *T) const {
    return getTypePath(T->key());
  }

  model::TypePath recordNewType(UpcastablePointer<Type> &&T);

  /// Uses `SortedVector::batch_insert()` to emplace all the elements from
  /// \ref NewTypes range into the `Types()` set.
  ///
  /// This inserts all the elements at the end of the underlying vector, and
  /// then triggers sorting, instead of conventional searching for the position
  /// of each element on its insertion.
  ///
  /// \note It takes advantage of `std::move_iterator` to ensure all
  ///       the elements are accessed strictly as r-values, so the original
  ///       container, \ref NewTypes range points to, is left in an unspecified
  ///       state after the invocation, as all of its elements are moved out of.
  ///
  /// \note Since it uses strict version of `batch-insert`'er, it will assert
  ///       if this causes multiple elements with the same key (\ref Type::Key)
  ///       to be present in the vector after the new elements are moved in.
  ///
  /// \tparam Range constrained input range type.
  /// \param  NewTypes the input range.
  template<range_with_value_type<UpcastablePointer<Type>> Range>
  void recordNewTypes(Range &&NewTypes) {
    auto Inserter = Types().batch_insert();

    static_assert(std::is_rvalue_reference_v<decltype(NewTypes)>);
    auto Movable = as_rvalue(std::move(NewTypes));
    for (UpcastablePointer<Type> &&NewType : Movable) {
      static_assert(std::is_rvalue_reference_v<decltype(NewType)>);
      Inserter.emplace(std::move(NewType));
    }
  }

  template<derived_from<model::Type> NewType, typename... ArgumentTypes>
  [[nodiscard]] std::pair<NewType &, model::TypePath>
  makeType(ArgumentTypes &&...Arguments) {
    using UT = model::UpcastableType;
    UT Result = UT::make<NewType>(std::forward<ArgumentTypes>(Arguments)...);
    model::TypePath ResultPath = recordNewType(std::move(Result));
    return { *llvm::cast<NewType>(ResultPath.get()), ResultPath };
  }

  model::TypePath getPrimitiveType(PrimitiveTypeKind::Values V,
                                   uint8_t ByteSize);

  model::TypePath getPrimitiveType(PrimitiveTypeKind::Values V,
                                   uint8_t ByteSize) const;

  bool verifyTypes() const debug_function;
  bool verifyTypes(bool Assert) const debug_function;
  bool verifyTypes(VerifyHelper &VH) const;

public:
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  bool verify() const;
  void dump() const debug_function;
  void dumpTypeGraph(const char *Path) const debug_function;
  std::string toString() const debug_function;
};

inline model::TypePath
getPrototype(const model::Binary &Binary,
             const model::DynamicFunction &DynamicFunction) {
  if (DynamicFunction.Prototype().isValid())
    return DynamicFunction.Prototype();
  else
    return Binary.DefaultPrototype();
}

#include "revng/Model/Generated/Late/Binary.h"
