#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerFactory.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

namespace pipeline {

/// A map from containerName to the container itself, which is initially empty.
///
/// This class is used mostly in two points, as the backing containers of each
/// step, as well as the containers containing the target extracted from each
/// step to perform the computation.
///
/// This class contains both the containers and a pointer to a factory that is
/// used to create that container when it does not exists.
class ContainerSet {
private:
  using Map = llvm::StringMap<std::unique_ptr<ContainerBase>>;

public:
  using const_iterator = Map::const_iterator;
  using iterator = Map::iterator;
  using value_type = Map::value_type;

private:
  Map Content;
  llvm::StringMap<const ContainerFactory *> Factories;

public:
  ContainerSet() = default;

  ContainerSet(Map Content) : Content(std::move(Content)) {}

  ContainerSet(ContainerSet &&) = default;
  ContainerSet &operator=(ContainerSet &&) = default;

  ContainerSet(const ContainerSet &) = delete;
  ContainerSet &operator=(const ContainerSet &) = delete;

  ~ContainerSet() = default;

public:
  const_iterator begin() const { return Content.begin(); }
  const_iterator end() const { return Content.end(); }

  iterator begin() { return Content.begin(); }
  iterator end() { return Content.end(); }

  iterator find(llvm::StringRef Name) { return Content.find(Name); }

  size_t size() const { return Factories.size(); }

public:
  ContainerSet cloneFiltered(const ContainerToTargetsMap &Targets);

  void mergeBack(ContainerSet &&Other) {
    for (auto &Entry : Other.Content) {
      revng_assert(containsOrCanCreate(Entry.first()));

      auto &LContainer = Content.find(Entry.first())->second;
      auto &RContainer = Entry.second;
      if (RContainer == nullptr)
        continue;

      if (LContainer == nullptr)
        LContainer = std::move(RContainer);
      else
        LContainer->mergeBack(std::move(*RContainer));
    }
  }

  ContainerBase &operator[](llvm::StringRef Name) {
    revng_assert(containsOrCanCreate(Name));
    if (Content[Name] == nullptr)
      Content[Name] = (*Factories[Name])(Name);
    auto &Pointer = Content.find(Name)->second;
    revng_assert(Pointer != nullptr);
    return *Pointer;
  }

  ContainerBase &at(llvm::StringRef Name) {
    revng_assert(contains(Name));
    return *Content.find(Name)->second;
  }

  const ContainerBase &at(llvm::StringRef Name) const {
    revng_assert(contains(Name));
    return *Content.find(Name)->second;
  }

  bool isContainerRegistered(llvm::StringRef Name) const {
    return Factories.find(Name) != Factories.end();
  }

  bool contains(llvm::StringRef Name) const {
    auto Iterator = Content.find(Name);
    return Iterator != Content.end() and Iterator->second != nullptr;
  }

  bool containsOrCanCreate(llvm::StringRef Name) const {
    return Content.find(Name) != Content.end();
  }

  template<typename T>
  const T &get(llvm::StringRef Name) const {
    return llvm::cast<T>(*Content.find(Name)->second);
  }

  template<typename T>
  T &get(llvm::StringRef Name) {
    return llvm::cast<T>(*Content.find(Name)->second);
  }

  template<typename T>
  T &getOrCreate(llvm::StringRef Name) {
    return llvm::cast<T>(operator[](Name));
  }

  bool contains(const Target &Target) const;

  ContainerToTargetsMap enumerate() const;

  llvm::Error verify() const;

public:
  void add(llvm::StringRef Name,
           const ContainerFactory &Factory,
           std::unique_ptr<ContainerBase> Container = nullptr) {
    Content.try_emplace(Name, std::move(Container));
    Factories.try_emplace(Name, &Factory);
  }

  llvm::Error remove(const ContainerToTargetsMap &ToRemove);
  void intersect(ContainerToTargetsMap &ToIntersect) const;

public:
  llvm::Error store(const revng::DirectoryPath &DirectoryPath) const;
  llvm::Error load(const revng::DirectoryPath &DirectoryPath);

  std::vector<revng::FilePath>
  getWrittenFiles(const revng::DirectoryPath &DirectoryPath) const;

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    for (const auto &Entry : Content) {
      indent(OS, Indentation);
      OS << Entry.first().str() << "\n";
      if (Entry.second != nullptr)
        Entry.second->enumerate().dump(OS, Indentation);
    }
  }

  void dump() const debug_function { dump(dbg); }
};

} // namespace pipeline
