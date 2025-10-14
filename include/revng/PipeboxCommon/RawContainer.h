#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline {

namespace detail {

template<Kind TheKind, ConstexprString TheName, ConstexprString TheMime>
class RawContainer {
public:
  static constexpr llvm::StringRef Name = TheName;
  static constexpr Kind Kind = TheKind;
  static constexpr llvm::StringRef MimeType = TheMime;

private:
  std::map<ObjectID, Buffer> Map;

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Map) | revng::to<std::set<ObjectID>>();
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    for (const auto &[Key, Value] : Data) {
      revng_assert(Key->kind() == Kind);
      // TODO: investigate later if ownership can be passed from the caller
      Map[*Key] = Buffer{ Value };
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects) {
      // TODO: investigate later if ownership can be passed to the caller
      Result[*Object] = Map.at(*Object);
    }
    return Result;
  }

  bool verify() const { return true; }

public:
  bool contains(const ObjectID &Key) const { return Map.contains(Key); }

  std::unique_ptr<llvm::raw_ostream> getOStream(const ObjectID &Key) {
    revng_assert(Key.kind() == Kind);
    revng_assert(not Map.contains(Key));
    return std::make_unique<llvm::raw_svector_ostream>(Map[Key].data());
  }

  std::unique_ptr<llvm::MemoryBuffer>
  getMemoryBuffer(const ObjectID &Key) const {
    const Buffer &TheBuffer = Map.at(Key);
    llvm::StringRef Ref(TheBuffer.data().data(), TheBuffer.data().size());
    return llvm::MemoryBuffer::getMemBuffer(Ref, "", false);
  }
};

template<Kind K, ConstexprString S, ConstexprString S2>
using RC = RawContainer<K, S, S2>;

constexpr auto TD = Kinds::TypeDefinition;

} // namespace detail

template<ConstexprString Name, ConstexprString Mime>
using BytesContainer = detail::RC<Kinds::Binary, Name, Mime>;

template<ConstexprString Name, ConstexprString Mime>
using FunctionToBytesContainer = detail::RC<Kinds::Function, Name, Mime>;

template<ConstexprString Name, ConstexprString Mime>
using TypeDefinitionToBytesContainer = detail::RC<detail::TD, Name, Mime>;

} // namespace revng::pypeline
