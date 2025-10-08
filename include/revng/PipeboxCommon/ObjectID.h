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

struct Kinds;
class ObjectID;

class Kind {
private:
  enum class ValueType : uint8_t {
    Binary = 0,
    Function,
    TypeDefinition,
  };
  friend ObjectID;

public:
  ValueType Value;

private:
  constexpr Kind(ValueType Value) : Value(Value){};
  friend std::hash<const Kind>;
  friend Kinds;

public:
  static std::vector<Kind> kinds() {
    return { Kind{ ValueType::Binary },
             Kind{ ValueType::Function },
             Kind{ ValueType::TypeDefinition } };
  };

  std::optional<Kind> parent() {
    if (Value == ValueType::Binary)
      return std::nullopt;
    else if (Value == ValueType::Function or Value == ValueType::TypeDefinition)
      return Kind{ ValueType::Binary };
    else
      revng_abort();
  }

  static Kind deserialize(llvm::StringRef Value) {
    if (Value == "binary") {
      return Kind{ ValueType::Binary };
    } else if (Value == "function") {
      return Kind{ ValueType::Function };
    } else if (Value == "type-definition") {
      return Kind{ ValueType::TypeDefinition };
    } else {
      revng_abort();
    }
  }

  std::string serlialize() {
    if (Value == ValueType::Binary) {
      return "binary";
    } else if (Value == ValueType::Function) {
      return "function";
    } else if (Value == ValueType::TypeDefinition) {
      return "type-definition";
    } else {
      revng_abort();
    }
  }

  uint64_t byteSize() {
    if (Value == ValueType::Binary) {
      return 0;
    } else if (Value == ValueType::Function) {
      return MetaAddress::BytesSize;
    } else if (Value == ValueType::TypeDefinition) {
      return packedSize<model::TypeDefinition::Key>;
    } else {
      revng_abort();
    }
  }

  std::strong_ordering operator<=>(const Kind &) const = default;
};

template<>
struct std::hash<const Kind> {
  uint64_t operator()(const Kind &Kind) const {
    return std::hash<Kind::ValueType>{}(Kind.Value);
  }
};

template<>
struct std::hash<Kind> : std::hash<const Kind> {};

struct Kinds {
  static inline constexpr Kind Binary{ Kind::ValueType::Binary };
  static inline constexpr Kind Function{ Kind::ValueType::Function };
  static inline constexpr Kind TypeDefinition{
    Kind::ValueType::TypeDefinition
  };
};

class ObjectID {
private:
  using TypeDefinitionKey = model::TypeDefinition::Key;
  using KeyType = std::variant<std::monostate, MetaAddress, TypeDefinitionKey>;
  KeyType Key;

public:
  // Create a root ObjectID
  ObjectID() = default;
  // Create a Function ObjectID
  ObjectID(const MetaAddress &Addr) : Key(Addr) {}
  // Create a TypeDefinition ObjectID
  ObjectID(const TypeDefinitionKey &Key) : Key(Key) {}

  Kind kind() const {
    auto Visitor = []<typename T>(const T &) {
      if constexpr (std::is_same_v<T, std::monostate>)
        return Kinds::Binary;
      else if constexpr (std::is_same_v<T, MetaAddress>)
        return Kinds::Function;
      else if constexpr (std::is_same_v<T, TypeDefinitionKey>)
        return Kinds::TypeDefinition;
      else
        revng_abort();
    };
    return std::visit(Visitor, Key);
  }

  std::optional<ObjectID> parent() const {
    const Kind &TheKind = kind();
    if (TheKind == Kinds::Binary)
      return std::nullopt;
    else if (TheKind == Kinds::Function or TheKind == Kinds::TypeDefinition)
      return ObjectID();
    else
      revng_abort();
  }

  static ObjectID root() { return ObjectID(); }

  std::string serialize() const {
    using namespace revng::ranks;
    using pipeline::locationString;
    const Kind &TheKind = kind();

    if (TheKind == Kinds::Binary)
      return locationString(Binary);
    else if (TheKind == Kinds::Function)
      return locationString(Function, std::get<MetaAddress>(Key));
    else if (TheKind == Kinds::TypeDefinition)
      return locationString(TypeDefinition, std::get<TypeDefinitionKey>(Key));
    else
      revng_abort();
  }

  static llvm::Expected<ObjectID> deserialize(llvm::StringRef Input) {
    using namespace revng::ranks;
    using pipeline::locationFromString;

    if (Input == "/binary") {
      return ObjectID();
    } else if (Input.starts_with("/function/")) {
      auto MaybeKey = locationFromString(Function, Input);
      if (not MaybeKey.has_value())
        return revng::createError("Failed deserializing ObjectID");
      MetaAddress Key = std::get<0>(std::get<0>(MaybeKey->tuple()));
      return ObjectID(Key);
    } else if (Input.starts_with("/type-definition/")) {
      auto MaybeKey = locationFromString(TypeDefinition, Input);
      if (not MaybeKey.has_value())
        return revng::createError("Failed deserializing ObjectID");
      TypeDefinitionKey Key = std::get<0>(MaybeKey->tuple());
      return ObjectID(Key);
    } else {
      return revng::createError("Failed deserializing ObjectID");
    }
  }

private:
  static uint8_t kindByte(const Kind &TheKind) {
    return static_cast<uint8_t>(TheKind.Value);
  }

public:
  std::vector<uint8_t> toBytes() const {
    const Kind &TheKind = kind();
    std::vector<uint8_t> Result;

    if (TheKind == Kinds::Binary) {
      return {};
    } else if (TheKind == Kinds::Function) {
      Result.push_back(kindByte(TheKind));
      append(std::get<MetaAddress>(Key).toBytes(), Result);
      return Result;
    } else if (TheKind == Kinds::TypeDefinition) {
      Result.push_back(kindByte(TheKind));
      append(::toBytes(std::get<TypeDefinitionKey>(Key)), Result);
      return Result;
    } else {
      revng_abort();
    }
  }

  static llvm::Expected<ObjectID> fromBytes(llvm::ArrayRef<uint8_t> Bytes) {
    constexpr size_t TDKSize = packedSize<TypeDefinitionKey>;

    if (Bytes.empty()) {
      return ObjectID();
    } else if (Bytes.size() == (1 + MetaAddress::BytesSize)
               and Bytes[0] == kindByte(Kinds::Function)) {
      auto Key = MetaAddress::fromBytes({ &Bytes[1], MetaAddress::BytesSize });
      return ObjectID(Key);
    } else if (Bytes.size() == (1 + TDKSize)
               and Bytes[0] == kindByte(Kinds::TypeDefinition)) {
      TypeDefinitionKey Key;
      ::fromBytes({ &Bytes[1], TDKSize }, Key);
      return ObjectID(Key);
    } else {
      return revng::createError("Failed deserializing ObjectID");
    }
  }

  const KeyType &key() const { return Key; }

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
