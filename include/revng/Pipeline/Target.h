#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Kind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace pipeline {

class TargetsList;
class ContainerBase;
class Context;

/// A target is a triple of Kind, PathComponents, and Exactness used to
/// enumerate and transform the contents of a container.
///
/// The kind is used to tell apart objects that are conceptually different or
/// have a relationship of containment or extension.
/// The PathComponent list is used to tell apart objects belonging to the same
/// Kind
class Target {
private:
  using PathComponents = std::vector<std::string>;
  PathComponents Components;
  const Kind *K;

public:
  Target(PathComponents Components, const Kind &K) :
    Components(std::move(Components)), K(&K) {
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(std::string PathComponent, const Kind &K) :
    Components({ std::move(PathComponent) }), K(&K) {
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(std::initializer_list<std::string> Names, const Kind &K) : K(&K) {
    for (auto Name : Names)
      Components.emplace_back(Name);
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(llvm::ArrayRef<llvm::StringRef> Names, const Kind &K) : K(&K) {
    for (auto Name : Names) {
      Components.emplace_back(Name.str());
    }
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(const Kind &K) : K(&K) {
    revng_assert(this->Components.size() == getKind().depth());
  }

public:
  bool operator<(const Target &Other) const {
    auto Self = std::tie(K, Components);
    auto OtherSelf = std::tie(Other.K, Other.Components);
    return Self < OtherSelf;
  }

  int operator<=>(const Target &Other) const;

  bool operator==(const Target &Other) const { return (*this <=> Other) == 0; }

public:
  const Kind &getKind() const { return *K; }
  const PathComponents &getPathComponents() const { return Components; }

public:
  void setKind(const Kind &NewKind) { K = &NewKind; }

  llvm::Error verify(const ContainerBase &Container) const {
    return K->verify(Container, *this);
  }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const debug_function {
    indent(OS, Indentation);
    OS << '/';

    const auto ComponentToString = [](const std::string &Component) {
      return Component;
    };

    auto Path = llvm::join(llvm::map_range(Components, ComponentToString), "/");
    OS << Path;
    OS << ':' << K->name().str();
    OS << '\n';
  }

  std::string serialize() const;
  static llvm::Expected<Target>
  deserialize(Context &Ctx, const KindsRegistry &Dict, llvm::StringRef String);

  template<typename OStream>
  void dumpPathComponents(OStream &OS) const debug_function {
    OS << "/";
    for (const auto &Entry : Components) {
      OS << Entry;
      if (&Entry != &Components.back())
        OS << "/";
    }
    OS << ":" << K->name().str();
  }

  void dump() const debug_function { dump(dbg); }
};

class KindsRegistry;

llvm::Error parseTarget(const Context &Ctx,
                        llvm::StringRef AsString,
                        const KindsRegistry &Dict,
                        TargetsList &Out);

/// a sorted list of targets.
class TargetsList {
public:
  using List = llvm::SmallVector<Target, 4>;
  using iterator = List::iterator;
  using const_iterator = List::const_iterator;
  using value_type = List::value_type;
  using size_type = List::size_type;
  using reference = List::reference;
  using pointer = List::pointer;

private:
  List Contained;

public:
  TargetsList() = default;
  TargetsList(List C) : Contained(std::move(C)) {}
  static TargetsList allTargets(const Context &Ctx, const Kind &K) {
    TargetsList ToReturn;
    K.appendAllTargets(Ctx, ToReturn);
    return ToReturn;
  }

  bool operator==(const TargetsList &Other) const = default;
  bool operator!=(const TargetsList &Other) const = default;

public:
  const Target &operator[](size_t Index) const { return Contained[Index]; }
  Target &operator[](size_t Index) { return Contained[Index]; }

public:
  llvm::Error verify(const ContainerBase &Container) const {
    for (const Target &T : *this)
      if (auto Error = T.verify(Container); Error)
        return Error;
    return llvm::Error::success();
  }
  iterator begin() { return Contained.begin(); }
  iterator end() { return Contained.end(); }
  const_iterator begin() const { return Contained.begin(); }
  const_iterator end() const { return Contained.end(); }
  size_t size() const { return Contained.size(); }
  bool empty() const { return Contained.empty(); }

  Target &front() { return Contained.front(); }
  const Target &front() const { return Contained.front(); }

  bool contains(const Target &Target) const;

  bool contains(const TargetsList &Targets) const {
    return llvm::all_of(Targets, [this](const Target &Target) {
      return contains(Target);
    });
  }

  TargetsList filter(const Kind &K) const {
    List C;
    for (const auto &Target : Contained)
      if (&Target.getKind() == &K)
        C.push_back(Target);

    return TargetsList(std::move(C));
  }

public:
  template<typename... Args>
  void emplace_back(Args &&...A) {
    Contained.emplace_back(std::forward<Args>(A)...);
    llvm::sort(Contained);
    Contained.erase(unique(Contained.begin(), Contained.end()),
                    Contained.end());
  }

  void merge(const TargetsList &Other);

  void push_back(const Target &Target) {
    Contained.push_back(Target);
    llvm::sort(Contained);
    Contained.erase(unique(Contained.begin(), Contained.end()),
                    Contained.end());
  }

  template<typename... Args>
  auto erase(Args &&...A) {
    return Contained.erase(std::forward<Args>(A)...);
  }

  TargetsList intersect(const TargetsList &Other) const {
    TargetsList ToReturn;
    std::set_intersection(begin(),
                          end(),
                          Other.begin(),
                          Other.end(),
                          std::back_inserter(ToReturn.Contained));
    llvm::sort(ToReturn.Contained);
    return ToReturn;
  }

private:
  struct Comp {
    bool operator()(const Target &T, const Kind &K) const {
      return &T.getKind() < &K;
    }
    bool operator()(const Kind &K, const Target &T) const {
      return &K < &T.getKind();
    }
  };

public:
  llvm::iterator_range<iterator> filterByKind(const Kind &K) {
    auto [b, e] = std::equal_range(begin(), end(), K, Comp());
    return llvm::make_range(b, e);
  }

  llvm::iterator_range<const_iterator> filterByKind(const Kind &K) const {
    auto [b, e] = std::equal_range(begin(), end(), K, Comp());
    return llvm::make_range(b, e);
  }

  llvm::SmallVector<const Kind *, 4> getContainedKinds() const {
    llvm::SmallVector<const Kind *, 4> ToReturn;

    auto Current = begin();
    while (Current != end()) {
      ToReturn.push_back(&Current->getKind());
      Current = std::upper_bound(Current, end(), Current->getKind(), Comp());
    }

    return ToReturn;
  }

public:
  void dump() const debug_function { dump(dbg); }

  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    for (const auto &Entry : Contained)
      Entry.dump(OS, Indentation);
  }
};

/// A map from container name to target list that is usually used to represents
/// the state of a step.
class ContainerToTargetsMap {
public:
  using Map = llvm::StringMap<TargetsList>;
  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;
  using value_type = Map::value_type;
  using key_iterator = decltype(declval<Map>().keys());

private:
  Map Status;

public:
  iterator begin() { return Status.begin(); }
  iterator end() { return Status.end(); }
  const_iterator begin() const { return Status.begin(); }
  const_iterator end() const { return Status.end(); }
  key_iterator keys() const { return Status.keys(); }

  bool empty() const { return targetsCount() == 0; }

  bool contains(llvm::StringRef ContainerName) const {
    return Status.find(ContainerName) != Status.end();
  }

  bool contains(const ContainerToTargetsMap &Other) const {
    for (const value_type &Pair : Other.Status) {
      if (Pair.second.empty())
        continue;

      if (not contains(Pair.first()))
        return false;

      if (not find(Pair.first())->second.contains(Pair.second))
        return false;
    }

    return true;
  }

  iterator find(llvm::StringRef ContainerName) {
    return Status.find(ContainerName);
  }

  const_iterator find(llvm::StringRef ContainerName) const {
    return Status.find(ContainerName);
  }

  TargetsList &at(llvm::StringRef Name) {
    revng_assert(Status.find(Name) != Status.end());
    return Status.find(Name)->getValue();
  }

  const TargetsList &at(llvm::StringRef ContainerName) const {
    revng_assert(Status.find(ContainerName) != Status.end());
    return Status.find(ContainerName)->second;
  }

  TargetsList &operator[](llvm::StringRef ContainerName) {
    return Status[ContainerName];
  }

public:
  void merge(const ContainerToTargetsMap &Other);

  void add(llvm::StringRef Name,
           std::initializer_list<std::string> Names,
           const Kind &K) {
    Status[Name].emplace_back(Names, K);
  }

  void add(llvm::StringRef Name, Target Target) {
    Status[Name].emplace_back(std::move(Target));
  }

  void add(llvm::StringRef Name, const TargetsList &Targets) {
    for (const auto &Target : Targets)
      Status[Name].emplace_back(Target);
  }

public:
  template<typename OStream>
  void
  dump(OStream &OS, size_t Indentation = 0, bool Parenthesis = true) const {
    if (Parenthesis) {
      indent(OS, Indentation);
      OS << "{\n";
    }
    auto ExtraIndentation = Indentation + (Parenthesis ? 1 : 0);
    for (const auto &Container : Status) {
      indent(OS, ExtraIndentation);
      OS << Container.first().str() << ":\n";
      Container.second.dump(OS, ExtraIndentation + 1);
    }
    if (Parenthesis) {
      indent(OS, Indentation);
      OS << "}\n";
    }
  }

  void dump() const debug_function { dump(dbg); }

private:
  size_t targetsCount() const {
    size_t Size = 0;
    for (const auto &Container : Status)
      Size += Container.second.size();
    return Size;
  }
};

using InvalidationMap = llvm::StringMap<ContainerToTargetsMap>;

llvm::Error parseTarget(const Context &Ctx,
                        ContainerToTargetsMap &CurrentStatus,
                        llvm::StringRef AsString,
                        const KindsRegistry &Dict);

inline void merge(InvalidationMap &Map, const InvalidationMap &Other) {
  for (const auto &Entry : Other) {
    if (auto Iter = Map.find(Entry.first()); Iter != Map.end()) {
      Iter->second.merge(Entry.second);
    } else {
      Map.try_emplace(Entry.first(), Entry.second);
    }
  }
}

template<typename T>
void prettyPrintStatus(const ContainerToTargetsMap &Targets,
                       T &OS,
                       size_t Indentation = 0) {
  for (const auto &Pair : Targets) {
    const auto &Name = Pair.first();
    const auto &List = Pair.second;
    if (List.empty())
      continue;

    indent(OS, Indentation);
    OS << Name << ":\n";

    for (const auto &Target : List)
      Target.dump(OS, Indentation + 1);
  }
}

} // namespace pipeline
