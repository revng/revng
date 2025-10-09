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
#include "revng/Model/Architecture.h"
#include "revng/Model/Configuration.h"
#include "revng/Model/DefinedType.h"
#include "revng/Model/DynamicFunction.h"
#include "revng/Model/EnumDefinition.h"
#include "revng/Model/Function.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/Register.h"
#include "revng/Model/Segment.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/CommonOptions.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/MetaAddressRangeSet.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "revng/Model/Generated/Early/Binary.h"

namespace model {
class VerifyHelper;
}

// TODO: Prevent changing the keys. Currently we need them to be public and
//       non-const for serialization purposes.

class model::Binary : public model::generated::Binary {
public:
  using generated::Binary::Binary;

public:
  /// Introduce a new type definition to the binary.
  ///
  /// \note there are also helpers for each of the type definition kinds which
  ///       should be preferred when creating a definition of a known kind.
  ///       as in, prefer `makeStructDefinition` to `makeDefinition<StructDef>`.
  ///
  /// \param Arguments A variadic argument list to pass to the type constructor.
  ///
  /// \tparam NewType The type of the new definition to make.
  ///
  /// \returns A pair of:
  ///          - the reference to the newly made definition which can be used
  ///            to modify it right away,
  ///          - the corresponding defined type ready to be attached to others.
  template<derived_from<model::TypeDefinition> NewType, typename... ArgTs>
  [[nodiscard]] std::pair<NewType &, model::UpcastableType>
  makeTypeDefinition(ArgTs &&...Args) {
    using GB = generated::Binary;
    auto &New = GB::makeTypeDefinition<NewType>(std::forward<ArgTs>(Args)...);
    return { New, makeType(New.key()) };
  }

  template<typename... Ts>
  [[nodiscard]] auto makeStructDefinition(Ts &&...As) {
    return makeTypeDefinition<StructDefinition>(std::forward<Ts>(As)...);
  }
  template<typename... Ts>
  [[nodiscard]] auto makeUnionDefinition(Ts &&...As) {
    return makeTypeDefinition<UnionDefinition>(std::forward<Ts>(As)...);
  }
  template<typename... Ts>
  [[nodiscard]] auto makeEnumDefinition(Ts &&...As) {
    return makeTypeDefinition<EnumDefinition>(std::forward<Ts>(As)...);
  }
  template<typename... Ts>
  [[nodiscard]] auto makeTypedefDefinition(Ts &&...As) {
    return makeTypeDefinition<TypedefDefinition>(std::forward<Ts>(As)...);
  }
  template<typename... Ts>
  [[nodiscard]] auto makeCABIFunctionDefinition(Ts &&...As) {
    return makeTypeDefinition<CABIFunctionDefinition>(std::forward<Ts>(As)...);
  }
  template<typename... Ts>
  [[nodiscard]] auto makeRawFunctionDefinition(Ts &&...As) {
    return makeTypeDefinition<RawFunctionDefinition>(std::forward<Ts>(As)...);
  }

public:
  /// Record the new type into the model and assign a new ID.
  ///
  /// \returns A pair of:
  ///          - the reference to the newly inserted definition which can be
  ///            used to modify it right away,
  ///          - the corresponding defined type ready to be attached to others.
  std::pair<TypeDefinition &, model::UpcastableType>
  recordNewType(model::UpcastableTypeDefinition &&T) {
    model::TypeDefinition &Result = recordNewTypeDefinition(std::move(T));
    return { Result, makeType(Result.key()) };
  }

  /// Uses `SortedVector::batch_insert()` to emplace all the elements from
  /// \ref NewTypes range into the `TypeDefinitions()` set.
  ///
  /// This inserts all the elements at the end of the underlying vector, and
  /// then triggers sorting, instead of conventional searching for the position
  /// of each element on its insertion.
  ///
  /// \note Unlike recordNewType, this method does not assign type IDs.
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
  template<RangeOf<UpcastablePointer<TypeDefinition>> Range>
  void recordNewTypeDefinitions(Range &&NewTypes) {
    auto Inserter = TypeDefinitions().batch_insert();

    static_assert(std::is_rvalue_reference_v<decltype(NewTypes)>);
    auto Movable = as_rvalue(std::move(NewTypes));
    for (UpcastablePointer<TypeDefinition> &&NewType : Movable) {
      static_assert(std::is_rvalue_reference_v<decltype(NewType)>);
      Inserter.emplace(std::move(NewType));
    }
  }

public:
  /// \note Only use this when absolutely necessary, for example, when doing
  ///       bulk reference replacement.
  ///       In the general case prefer \ref makeType instead.
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

  model::UpcastableType makeType(const model::TypeDefinition::Key &Key) {
    return model::DefinedType::make(getDefinitionReference(Key));
  }
  model::UpcastableType makeConstType(const model::TypeDefinition::Key &Key) {
    return model::DefinedType::makeConst(getDefinitionReference(Key));
  }
  model::UpcastableType makeType(const model::TypeDefinition::Key &Key) const {
    return model::DefinedType::make(getDefinitionReference(Key));
  }
  model::UpcastableType
  makeConstType(const model::TypeDefinition::Key &Key) const {
    return model::DefinedType::makeConst(getDefinitionReference(Key));
  }

public:
  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref DefaultPrototype() when you need to assign a new one.
  model::TypeDefinition *defaultPrototype() {
    // TODO: after `abi::Definition` is merged back into the model,
    //       the prototype will always be present, so this should return
    //       a reference instead.
    if (DefaultPrototype().isEmpty())
      return nullptr;
    else
      return &DefaultPrototype()->toPrototype();
  }

  /// The helper for the prototype unwrapping.
  /// Use this when you need to access/modify the existing prototype,
  /// and \ref DefaultPrototype() when you need to assign a new one.
  const model::TypeDefinition *defaultPrototype() const {
    // TODO: after `abi::Definition` is merged back into the model,
    //       the prototype will always be present, so this should return
    //       a reference instead.
    if (DefaultPrototype().isEmpty())
      return nullptr;
    else
      return &DefaultPrototype()->toPrototype();
  }

  model::TypeDefinition *prototypeOrDefault(model::TypeDefinition *Prototype) {
    if (Prototype)
      return Prototype;

    return defaultPrototype();
  }

  const model::TypeDefinition *
  prototypeOrDefault(const model::TypeDefinition *Prototype) const {
    if (Prototype)
      return Prototype;

    return defaultPrototype();
  }

public:
  bool verify(VerifyHelper &VH) const;
  bool verify(bool Assert) const debug_function;
  bool verify() const debug_function;

  bool verifyTypeDefinitions(VerifyHelper &VH) const;
  bool verifyTypeDefinitions(bool Assert) const debug_function;
  bool verifyTypeDefinitions() const debug_function;

public:
  void dumpTypeGraph(const char *Path) const debug_function;

public:
  MetaAddressRangeSet executableRanges() const;
};

#include "revng/Model/Generated/Late/Binary.h"
