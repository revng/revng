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
    FirstAvailableID(Binary.getAvailableTypeID()),
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
    revng_assert(Binary.getAvailableTypeID() == FirstAvailableID,
                 "Unable to commit: owner's id requirements have changed");

    Binary.recordNewTypeDefinitions(std::move(Definitions));
    NextAvailableID = FirstAvailableID = Binary.getAvailableTypeID();
    Definitions.clear();
  }

  /// Aborts dependency management and discards any modification to
  /// the managed model made in the \ref Types vector.
  ///
  /// The managed model is the one \ref Binary points to.
  void drop() {
    NextAvailableID = FirstAvailableID;
    Definitions.clear();
  }

  [[nodiscard]] bool empty() const { return Definitions.empty(); }

public:
  /// A helper for new type creation.
  ///
  /// Its usage mirrors that of `model::Binary::makeTypeDefinition<NewType>()`.
  ///
  /// \note This function forcefully assign a new type ID.
  template<typename NewD, typename... Ts>
  [[nodiscard]] std::pair<NewD &, model::DefinitionReference>
  makeTypeDefinition(Ts &&...As) {
    using UT = model::UpcastableTypeDefinition;
    auto &D = Definitions.emplace_back(UT::make<NewD>(std::forward<Ts>(As)...));
    NewD *Upcasted = llvm::cast<NewD>(D.get());

    // Assign the ID
    revng_assert(Upcasted->ID() == 0);
    Upcasted->ID() = NextAvailableID++;

    auto ResultPath = Binary.getDefinitionReference(Upcasted->key());
    return { *Upcasted, ResultPath };
  }

public:
  /// A helper for primitive type selection when binary access is limited.
  ///
  /// It mirrors `model::Binary::getPrimitiveType`.
  inline model::DefinitionReference
  getPrimitiveType(model::PrimitiveKind::Values Kind, uint8_t Size) {
    return Binary.getPrimitiveType(Kind, Size);
  }

  /// A helper streamlining selection of types for registers.
  ///
  /// \param Register Any CPU register the model is aware of.
  ///
  /// \return A primitive type in \ref Binary.
  inline model::DefinitionReference
  defaultRegisterType(model::Register::Values Register) {
    return getPrimitiveType(model::Register::primitiveKind(Register),
                            model::Register::getSize(Register));
  }

  /// A helper streamlining selection of types for registers.
  ///
  /// \param Register Any CPU register the model is aware of.
  ///
  /// \return A primitive type in \ref Binary.
  inline model::DefinitionReference
  genericRegisterType(model::Register::Values Register) {
    return getPrimitiveType(model::PrimitiveKind::Generic,
                            model::Register::getSize(Register));
  }
};

} // namespace model
