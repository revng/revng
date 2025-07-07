#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/StringRef.h"

#include "revng/Model/TypeDefinition.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/MetaAddress.h"

class ObjectID {
private:
  using TypeDefinitionKey = model::TypeDefinition::Key;
  std::variant<std::monostate, MetaAddress, TypeDefinitionKey> Key;

public:
  enum class Relation {
    // Relationship between `ObjectID::Kind`s. Must be kept ins sync with the
    // ObjectID.Relation in Python
    SAME = 0,
    ANCESTOR = 1,
    DESCENDANT = 2,
    UNRELATED = 3,
  };

  enum class Kind {
    Root = 0,
    Function,
    TypeDefinition,
  };

  static Relation kindRelation(Kind From, Kind To) {
    if (From == To)
      return Relation::SAME;
    if (From == Kind::Root)
      return Relation::ANCESTOR;
    if ((From == Kind::Function and To == Kind::TypeDefinition)
        or (From == Kind::TypeDefinition and To == Kind::Function))
      return Relation::UNRELATED;
    if ((From == Kind::Function or From == Kind::TypeDefinition)
        and To == Kind::Root)
      return Relation::DESCENDANT;
    revng_abort();
  }

public:
  // Create a root ObjectID
  ObjectID() = default;
  // Create a Function ObjectID
  ObjectID(const MetaAddress &Addr) : Key(Addr) {}
  // Create a TypeDefinition ObjectID
  ObjectID(const TypeDefinitionKey &Key) : Key(Key) {}

  Kind kind() const {
    if (std::holds_alternative<std::monostate>(Key)) {
      return Kind::Root;
    } else if (std::holds_alternative<MetaAddress>(Key)) {
      return Kind::Function;
    } else if (std::holds_alternative<TypeDefinitionKey>(Key)) {
      return Kind::TypeDefinition;
    } else {
      revng_abort();
    }
  }

  std::optional<ObjectID> parent() const {
    switch (kind()) {
    case Kind::Root:
      return std::nullopt;
    case Kind::Function:
      return ObjectID();
    case Kind::TypeDefinition:
      return ObjectID();
    default:
      revng_abort();
    }
  }

  std::string serialize() const {
    namespace r = revng::ranks;
    using pipeline::locationString;

    switch (kind()) {
    case Kind::Root:
      return "/root/";
    case Kind::Function:
      return locationString(r::Function, std::get<MetaAddress>(Key));
    case Kind::TypeDefinition:
      return locationString(r::TypeDefinition,
                            std::get<TypeDefinitionKey>(Key));
    default:
      revng_abort();
    }
  }

  llvm::Error deserialize(llvm::StringRef Input) {
    namespace r = revng::ranks;
    using pipeline::locationFromString;

    if (Input == "/root/") {
      Key = std::monostate{};
    } else if (Input.starts_with("/function/")) {
      auto MaybeKey = locationFromString(r::Function, Input);
      if (not MaybeKey.has_value())
        return revng::createError("Failed deserializing ObjectID");
      Key = std::get<0>(std::get<0>(MaybeKey->tuple()));
    } else if (Input.starts_with("/type-definition/")) {
      auto MaybeKey = locationFromString(r::TypeDefinition, Input);
      if (not MaybeKey.has_value())
        return revng::createError("Failed deserializing ObjectID");
      Key = std::get<0>(MaybeKey->tuple());
    } else {
      return revng::createError("Failed deserializing ObjectID");
    }
    return llvm::Error::success();
  }

  std::strong_ordering operator<=>(const ObjectID &) const = default;
  friend std::hash<const ObjectID>;
};

template<>
struct std::hash<const ObjectID> {
  uint64_t operator()(const ObjectID &Obj) const {
    return std::hash<decltype(Obj.Key)>{}(Obj.Key);
  }
};

template<>
struct std::hash<ObjectID> : std::hash<const ObjectID> {};
