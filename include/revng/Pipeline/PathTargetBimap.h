#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Target.h"
#include "revng/TupleTree/TupleTreePath.h"

namespace pipeline {
class Context;

class TargetInContainer {
private:
  Target Content;
  std::string ContainerName;

public:
  TargetInContainer(const Target &Target, const std::string &ContainerName) :
    Content(Target), ContainerName(ContainerName) {}

  const Target &getTarget() const { return Content; }

  llvm::StringRef getContainerName() const {
    return llvm::StringRef(ContainerName);
  }

  Target &getTarget() { return Content; }

  std::string &getContainerName() { return ContainerName; }

  bool operator==(const TargetInContainer &) const = default;

  bool operator!=(const TargetInContainer &) const = default;

  bool operator<(const TargetInContainer &Other) const {
    const auto &LHS = std::tie(Content, ContainerName);
    return LHS < std::tie(Other.Content, Other.ContainerName);
  }

  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const debug_function {
    indent(OS, Indentation);
    OS << Content.toString() << " in " << ContainerName << "\n";
  }
};

#define VECTOR_BIMAP
#ifdef VECTOR_BIMAP
class PathTargetBimap {
private:
  // WIP: drop mutable
  mutable std::vector<std::pair<TupleTreePath, TargetInContainer>> Map;
  mutable bool Sorted = true;

public:
  explicit PathTargetBimap() {}

public:
  auto begin() { return Map.begin(); }
  auto end() { return Map.end(); }
  auto begin() const { return Map.begin(); }
  auto end() const { return Map.end(); }

  auto find(const TupleTreePath &Path) const {
    sort();
    auto StartIt = llvm::
      lower_bound(Map,
                  std::pair<TupleTreePath,
                            TargetInContainer>(Path,
                                               TargetInContainer(Target(),
                                                                 "")));
    auto EndIt = StartIt;
    while (EndIt != Map.end() and EndIt->first == Path)
      ++EndIt;
    return llvm::make_range(StartIt, EndIt);
  }

public:
  void merge(PathTargetBimap &&Other) {
    Sorted = false;
    for (auto &Entry : Other.Map)
      Map.emplace_back(std::move(Entry));
  }

public:
  void insert(const TargetInContainer &TargetInContainer,
              TupleTreePath &&Path) {
    Sorted = false;
    Map.emplace_back(std::move(Path), TargetInContainer);
  }

  void remove(const TargetsList &List, llvm::StringRef ContainerName) {
    // Note: erasing entries does not affect Sorted status
    llvm::erase_if(Map,
                   [&](const std::pair<TupleTreePath, TargetInContainer>
                         &Entry) {
                     return Entry.second.getContainerName() == ContainerName
                            and List.contains(Entry.second.getTarget());
                   });
  }

private:
  void sort() const {
    if (Sorted)
      return;

    Sorted = true;
    llvm::sort(Map);
  }
};

#else
/// A PathTargetBimap is a many to many relationship between a TupleTreePaths
/// and TargetInContainer. It can be queried both ways, so from a target and a
/// container you can get the list of tuple tree paths that contribuited to that
/// target in that container, and from a tuple tree path you can know all the
/// targets that have been created reading the field pointed by that path.
class PathTargetBimap {
private:
  using MapType = std::map<TupleTreePath, std::set<TargetInContainer>>;
  MapType Map;

public:
  explicit PathTargetBimap() : Map() {}

public:
  auto begin() { return Map.begin(); }
  auto end() { return Map.end(); }
  auto begin() const { return Map.begin(); }
  auto end() const { return Map.end(); }

  auto find(const TupleTreePath &Path) const { return Map.find(Path); }

public:
  void merge(PathTargetBimap &&Other) {
    for (auto &Entry : Other.Map)
      for (auto &Target : Entry.second)
        insert(std::move(Target), std::move(Entry.first));
  }

public:
  void insert(const TargetInContainer &TargetInContainer,
              const TupleTreePath &Path) {
    Map[Path].insert(TargetInContainer);
  }

  void insert(const Target &Target,
              const std::string &ContainerName,
              const TupleTreePath &Path) {
    TargetInContainer Located(Target, ContainerName);
    insert(Located, Path);
  }

  void remove(const TargetsList &List, llvm::StringRef ContainerName) {
    for (auto &[Path, Targets] : Map) {
      std::erase_if(Targets, [&List](const TargetInContainer &TIC) -> bool {
        return List.contains(TIC.getTarget());
      });
    }
  }
};
#endif

} // namespace pipeline
