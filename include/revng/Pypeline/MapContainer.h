#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/ObjectID.h"

namespace revng::pypeline {

namespace detail {

template<ObjectID::Kind KindParam, ConstexprString NameParam>
class MapContainer {
public:
  static constexpr llvm::StringRef Name = NameParam;
  static constexpr ObjectID::Kind Kind = KindParam;

private:
  std::map<ObjectID, Buffer> Map;

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Map) | revng::to<std::set<ObjectID>>();
  }

  void deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                     Data) {
    for (const auto &[Key, Value] : Data) {
      revng_assert(Key->kind() == Kind);
      Map[*Key] = Buffer{ Value };
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects)
      Result[*Object] = Map.at(*Object);
    return Result;
  }

  bool verify() const { return true; }

public:
  bool contains(ObjectID Key) const { return Map.contains(Key); }

  std::unique_ptr<llvm::raw_ostream> getOStream(ObjectID Key) {
    revng_assert(Key.kind() == Kind);
    revng_assert(not Map.contains(Key));
    return std::make_unique<llvm::raw_svector_ostream>(Map[Key].get());
  }

  std::unique_ptr<llvm::MemoryBuffer> getMemoryBuffer(ObjectID Key) const {
    const Buffer &TheBuffer = Map.at(Key);
    llvm::StringRef Ref(TheBuffer.ref().data(), TheBuffer.ref().size());
    return llvm::MemoryBuffer::getMemBuffer(Ref, "", false);
  }
};

} // namespace detail

using RootBuffer = detail::MapContainer<ObjectID::Kind::Root, "root-buffer">;

using FunctionMap = detail::MapContainer<ObjectID::Kind::Function,
                                         "function-map">;

using TypeDefinitionMap = detail::MapContainer<ObjectID::Kind::TypeDefinition,
                                               "type-definition-map">;

} // namespace revng::pypeline
