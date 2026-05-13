#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline {

namespace detail {

template<Kind TheKind>
class RawContainer {
public:
  static constexpr Kind Kind = TheKind;

private:
  bool Disposable = false;
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

  void setIsDisposable() { Disposable = true; }

  void disposeIfPossible() {
    if (Disposable)
      return;

    Map.clear();
    Disposable = false;
  }

public:
  bool contains(const ObjectID &Key) const { return Map.contains(Key); }

  std::unique_ptr<llvm::raw_pwrite_stream> getOStream(const ObjectID &Key) {
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

constexpr auto TD = Kinds::TypeDefinition;

} // namespace detail

using BytesContainer = detail::RawContainer<Kinds::Binary>;
using FunctionToBytesContainer = detail::RawContainer<Kinds::Function>;
using TypeDefinitionToBytesContainer = detail::RawContainer<detail::TD>;

} // namespace revng::pypeline
