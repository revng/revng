#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Global.h"
#include "revng/Storage/Path.h"

namespace pipeline {
class GlobalsMap {
private:
  using MapType = std::map<std::string, std::unique_ptr<Global>>;
  MapType Map;

public:
  DiffMap diff(const GlobalsMap &Other) const {
    DiffMap ToReturn;

    for (const auto &Pair : Map) {
      const auto &OtherPair = *Other.Map.find(Pair.first);

      auto Diff = Pair.second->diff(*OtherPair.second);
      ToReturn.try_emplace(Pair.first, std::move(Diff));
    }

    return ToReturn;
  }

private:
  static const Global *
  dereferenceIterator(const MapType::const_iterator::value_type &Pair) {
    return Pair.second.get();
  }

public:
  auto begin() const {
    return llvm::map_iterator(Map.begin(), dereferenceIterator);
  }

  auto end() const {
    return llvm::map_iterator(Map.end(), dereferenceIterator);
  }

  template<typename ToAdd, typename... T>
  void emplace(llvm::StringRef Name, T &&...Args) {
    Map.try_emplace(Name.str(),
                    std::make_unique<ToAdd>(Name, std::forward<T>(Args)...));
  }

  template<typename T>
  llvm::Expected<T *> get(llvm::StringRef Name) const {
    auto It = Map.find(Name.str());
    if (It == Map.end()) {
      auto *Message = "Unknown Global %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    auto *Casted = llvm::dyn_cast<T>(It->second.get());
    if (Casted == nullptr) {
      auto *Message = "requested to cast %s to the wrong type";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    return Casted;
  }

  llvm::Expected<pipeline::Global *> get(llvm::StringRef Name) const {
    auto It = Map.find(Name.str());
    if (It == Map.end()) {
      auto *Message = "Unknown Global %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }
    return &*It->second;
  }

  llvm::StringRef getName(size_t Index) const {
    return std::next(Map.begin(), Index)->first;
  }

  llvm::Error serialize(llvm::StringRef GlobalName,
                        llvm::raw_ostream &OS) const {
    auto MaybeGlobal = get(GlobalName);
    if (!MaybeGlobal)
      return MaybeGlobal.takeError();
    return MaybeGlobal.get()->serialize(OS);
  }

  llvm::Error deserialize(llvm::StringRef GlobalName,
                          const llvm::MemoryBuffer &Buffer) {
    auto MaybeGlobal = get(GlobalName);
    if (!MaybeGlobal)
      return MaybeGlobal.takeError();
    return MaybeGlobal.get()->fromString(Buffer.getBuffer());
  }

  llvm::Error verify(llvm::StringRef GlobalName) const {
    auto MaybeGlobal = get(GlobalName);
    if (!MaybeGlobal)
      return MaybeGlobal.takeError();
    if (not MaybeGlobal.get()->verify()) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not verify " + GlobalName);
    }
    return llvm::Error::success();
  }

  llvm::Expected<std::unique_ptr<Global>>
  createNew(llvm::StringRef GlobalName,
            const llvm::MemoryBuffer &Buffer) const {
    auto MaybeGlobal = get(GlobalName);
    if (!MaybeGlobal)
      return MaybeGlobal.takeError();
    return MaybeGlobal.get()->createNew(GlobalName, Buffer);
  }

  llvm::Expected<std::unique_ptr<Global>> clone(llvm::StringRef GlobalName) {
    auto MaybeGlobal = get(GlobalName);
    if (!MaybeGlobal)
      return MaybeGlobal.takeError();
    return MaybeGlobal.get()->clone();
  }

  llvm::Error store(const revng::DirectoryPath &Path) const;
  llvm::Error load(const revng::DirectoryPath &Path);

  size_t size() const { return Map.size(); }

public:
  GlobalsMap() = default;
  ~GlobalsMap() = default;
  GlobalsMap(GlobalsMap &&Other) = default;
  GlobalsMap(const GlobalsMap &Other) {
    for (const auto &Entry : Other.Map)
      Map.try_emplace(Entry.first, Entry.second->clone());
  }

  GlobalsMap &operator=(GlobalsMap &&Other) = default;
  GlobalsMap &operator=(const GlobalsMap &Other) {
    if (this == &Other)
      return *this;

    Map = MapType();

    for (const auto &Entry : Other.Map)
      Map.try_emplace(Entry.first, Entry.second->clone());

    return *this;
  }
  void collectReadFields(const TargetInContainer &Target,
                         llvm::StringMap<PathTargetBimap> &Out) const {
    for (const auto &Global : Map) {
      Global.second->collectReadFields(Target, Out[Global.first]);
    }
  }

  void clearAndResume() const {
    for (const auto &Global : Map)
      Global.second->clearAndResume();
  }
  void pushReadFields() const {
    for (const auto &Global : Map)
      Global.second->pushReadFields();
  }
  void popReadFields() const {
    for (const auto &Global : Map)
      Global.second->popReadFields();
  }
  void stopTracking() const {
    for (const auto &Global : Map)
      Global.second->stopTracking();
  }
};
} // namespace pipeline
