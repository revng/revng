#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

namespace model {

/// This is simple type definition container that allows keeping definitions
/// connected to a specific binary without inserting them in.
class TypeBucket {
private:
  llvm::SmallVector<model::UpcastableTypeDefinition, 16> Definitions;
  model::Binary &Binary;
  uint64_t FirstAvailableID;
  uint64_t NextAvailableID;

public:
  TypeBucket(model::Binary &Binary) :
    Binary(Binary),
    FirstAvailableID(Binary.getNextAvailableIDForTypeDefinitions()),
    NextAvailableID(FirstAvailableID) {}
  TypeBucket(const TypeBucket &Another) = delete;
  TypeBucket(TypeBucket &&Another) = default;
  ~TypeBucket() {
    revng_assert(Definitions.empty(),
                 "The bucket has to be explicitly committed or dropped.");
  }

  /// Marks all the current dependencies as essential and merges them into
  /// the managed model.
  ///
  /// The managed model is the one \ref Binary points to.
  void commit() {
    uint64_t ActualNextID = Binary.getNextAvailableIDForTypeDefinitions();
    revng_assert(ActualNextID == FirstAvailableID,
                 "Unable to commit: owner's id requirements have changed");

    Binary.recordNewTypeDefinitions(std::move(Definitions));

    ActualNextID = Binary.getNextAvailableIDForTypeDefinitions();
    NextAvailableID = FirstAvailableID = ActualNextID;
    Definitions.clear();
  }

  /// Aborts dependency management and discards any modification to
  /// the managed model made in the \ref Types vector.
  ///
  /// The managed model is the one \ref Binary points to.
  void drop() {
    uint64_t ActualNextID = Binary.getNextAvailableIDForTypeDefinitions();
    NextAvailableID = FirstAvailableID = ActualNextID;
    Definitions.clear();
  }

  [[nodiscard]] bool empty() const { return Definitions.empty(); }

public:
  /// A helper for new type creation.
  ///
  /// Its usage mirrors that of `model::Binary::makeTypeDefinition<NewType>()`.
  ///
  /// \note This function forcefully assigns a new type ID.
  template<typename NewD, typename... Ts>
  [[nodiscard]] std::pair<NewD &, model::UpcastableType>
  makeTypeDefinition(Ts &&...As) {
    using UT = model::UpcastableTypeDefinition;
    auto &D = Definitions.emplace_back(UT::make<NewD>(std::forward<Ts>(As)...));
    NewD *Upcasted = llvm::cast<NewD>(D.get());

    // Assign the ID
    revng_assert(Upcasted->ID() == uint64_t(-1));
    Upcasted->ID() = NextAvailableID++;

    return { *Upcasted, Binary.makeType(Upcasted->key()) };
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
};

} // namespace model
