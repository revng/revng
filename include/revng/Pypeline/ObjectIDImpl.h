#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <set>

#include "llvm/ADT/StringRef.h"

#include "revng/Model/TypeDefinition.h"
#include "revng/Support/MetaAddress.h"

class ObjectID {
private:
  using TypeKey = model::TypeDefinition::Key;
  std::variant<std::monostate, MetaAddress, TypeKey> Component;
  friend constexpr std::strong_ordering operator<=>(const ObjectID &,
                                                    const ObjectID &);

public:
  enum Kind {
    Root = 0,
    Function = 1,
    Type = 2
  };

public:
  static constexpr llvm::StringRef Name = "ObjectID";
  ObjectID() : Component() {}

  Kind kind() const {
    if (std::holds_alternative<std::monostate>(Component)) {
      return Kind::Root;
    } else if (std::holds_alternative<MetaAddress>(Component)) {
      return Kind::Function;
    } else if (std::holds_alternative<TypeKey>(Component)) {
      return Kind::Type;
    } else {
      revng_abort();
    }
  }

  std::vector<std::string> components() const {
    switch (kind()) {
    case Kind::Root:
      return {};
    case Kind::Function:
      return { std::get<MetaAddress>(Component).toString() };
    case Kind::Type:
      return { toString(std::get<TypeKey>(Component)) };
    }
  }

  optional<ObjectID> parent() const {
    switch (kind()) {
    case Kind::Root:
      return std::nullopt;
    case Kind::Function:
      return ObjectID();
    case Kind::Type:
      return ObjectID();
    }
  }

  std::string serialize() const {
    switch (kind()) {
    case Kind::Root:
      return "/root/";
    case Kind::Function:
      return "/function/" + std::get<MetaAddress>(Component).toString();
    case Kind::Type:
      return "/type/" + toString(std::get<TypeKey>(Component));
    }
  }

  bool deserialize(llvm::StringRef Input) {
    if (not Input.starts_with("/"))
      return false;

    llvm::SmallVector<llvm::StringRef, 3> Parts;
    Input.split(Parts, "/");
    if (Parts[1] == "root") {
      Component = std::monostate{};
      return true;
    } else if (Parts[1] == "function") {
      Component = MetaAddress::fromString(Parts[2]);
      return true;
    } else if (Parts[1] == "type") {
      llvm::Expected<TypeKey> MaybeKey = fromString<TypeKey>(Parts[2]);
      if (not MaybeKey) {
        llvm::consumeError(MaybeKey.takeError());
        return false;
      }
      Component = *MaybeKey;
      return true;
    } else {
      return false;
    }
  }
};

inline constexpr std::strong_ordering operator<=>(const ObjectID &LHS,
                                                  const ObjectID &RHS) {
  using Kind = ObjectID::Kind;
  Kind LKind = LHS.kind();
  Kind RKind = RHS.kind();
  if (LKind == RKind) {
    if (LKind == Kind::Root)
      return std::strong_ordering::equal;
    if (LKind == Kind::Function) {
      const MetaAddress &LAddr = std::get<MetaAddress>(LHS.Component);
      const MetaAddress &RAddr = std::get<MetaAddress>(RHS.Component);
      return LAddr <=> RAddr;
    }
    if (LKind == Kind::Type) {
      using TypeKey = ObjectID::TypeKey;
      const TypeKey &LKey = std::get<TypeKey>(LHS.Component);
      const TypeKey &RKey = std::get<TypeKey>(RHS.Component);
      return LKey <=> RKey;
    }
    revng_abort();
  } else {
    if (LKind == Kind::Root)
      return std::strong_ordering::greater;
    if (RKind == Kind::Root)
      return std::strong_ordering::less;
    if (LKind == Kind::Function)
      return std::strong_ordering::greater;
    if (LKind == Kind::Type)
      return std::strong_ordering::less;
    revng_abort();
  }
}
