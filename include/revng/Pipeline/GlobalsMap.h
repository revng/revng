#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Global.h"

namespace pipeline {
class GlobalsMap {
private:
  using MapType = llvm::StringMap<std::unique_ptr<Global>>;
  MapType Map;

public:
  DiffMap diff(const GlobalsMap &Other) const {
    DiffMap ToReturn;

    for (const auto &Pair : Map) {
      const auto &OtherPair = *Other.Map.find(Pair.first());

      auto Diff = Pair.second->diff(*OtherPair.second);
      ToReturn.try_emplace(Pair.first(), std::move(Diff));
    }

    return ToReturn;
  }

  template<typename ToAdd, typename... T>
  void emplace(llvm::StringRef Name, T &&...Args) {
    Map.try_emplace(Name, std::make_unique<ToAdd>(std::forward<T>(Args)...));
  }

  template<typename T>
  llvm::Expected<T *> get(llvm::StringRef Name) const {
    auto Iter = Map.find(Name);
    if (Iter == Map.end()) {
      auto *Message = "could not find %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    auto *Casted = llvm::dyn_cast<T>(Iter->second.get());
    if (Casted == nullptr) {
      auto *Message = "requested to cast %s to the wrong type";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    return Casted;
  }

  llvm::StringRef getName(size_t Index) const {
    return std::next(Map.begin(), Index)->first();
  }

  llvm::Error
  serialize(llvm::StringRef GlobalName, llvm::raw_ostream &OS) const {
    auto Iter = Map.find(GlobalName);
    if (Iter == Map.end()) {
      auto *Message = "pipeline loader context did not contained object %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     GlobalName.str().c_str());
    }

    return Iter->second->serialize(OS);
  }

  llvm::Error
  deserialize(llvm::StringRef GlobalName, const llvm::MemoryBuffer &Buffer) {
    auto Iter = Map.find(GlobalName);
    if (Iter == Map.end()) {
      auto *Message = "pipeline loader context did not contained object %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     GlobalName.str().c_str());
    }

    return Iter->second->deserialize(Buffer);
  }

  llvm::Error storeToDisk(llvm::StringRef Path) const;
  llvm::Error loadFromDisk(llvm::StringRef Path);

  size_t size() const { return Map.size(); }

public:
  GlobalsMap() = default;
  ~GlobalsMap() = default;
  GlobalsMap(GlobalsMap &&Other) = default;
  GlobalsMap(const GlobalsMap &Other) {
    for (const auto &Entry : Other.Map)
      Map.try_emplace(Entry.first(), Entry.second->clone());
  }

  GlobalsMap &operator=(GlobalsMap &&Other) = default;
  GlobalsMap &operator=(const GlobalsMap &Other) {
    if (this == &Other)
      return *this;

    Map = MapType();

    for (const auto &Entry : Other.Map)
      Map.try_emplace(Entry.first(), Entry.second->clone());

    return *this;
  }
};
} // namespace pipeline
