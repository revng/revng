#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline {

template<TupleTreeCompatible T, Kind TheKind>
class TupleTreeContainer {
public:
  static constexpr Kind Kind = TheKind;
  static constexpr llvm::StringRef MimeType = "text/x.yaml";

private:
  bool Disposable = false;
  std::map<ObjectID, TupleTree<T>> Map;

public:
  std::set<ObjectID> objects() const {
    return std::views::keys(Map) | revng::to<std::set<ObjectID>>();
  }

  void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<char>> Data) {
    for (const auto &[Key, Value] : Data) {
      revng_assert(Key->kind() == Kind);
      llvm::StringRef String{ Value.data(), Value.size() };
      Map[*Key] = llvm::cantFail(TupleTree<T>::fromString(String));
    }
  }

  std::map<ObjectID, Buffer>
  serialize(const std::vector<const ObjectID *> Objects) const {
    std::map<ObjectID, Buffer> Result;
    for (const ObjectID *Object : Objects) {
      llvm::raw_svector_ostream OS(Result[*Object].data());
      Map.at(*Object).serialize(OS);
    }
    return Result;
  }

  bool verify() const { return true; }

  void setIsDisposable() { Disposable = true; }

  void disposeIfPossible() {
    if (not Disposable)
      return;
    Map.clear();
    Disposable = false;
  }

public:
  bool contains(const ObjectID &Key) const { return Map.contains(Key); }
  TupleTree<T> &getElement(const ObjectID &Key) { return Map[Key]; }
  const TupleTree<T> &getElement(const ObjectID &Key) const {
    return Map.at(Key);
  }
};

} // namespace revng::pypeline
