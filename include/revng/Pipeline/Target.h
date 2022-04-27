#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace pipeline {

namespace Exactness {
enum Values { Exact, DerivedFrom };
}

class TargetsList;
class ContainerBase;

/// A target is a triple of Kind, PathComponents, and Exactness used to
/// enumerate and transform the contents of a container.
///
/// The kind is used to tell apart objects that are conceptually different or
/// have a relationship of containment or extension.
/// The PathComponent list is used to tell apart objects belonging to the same
/// Kind The Exactness express that a requirements can be satisfied by a kind or
/// its extension.
class Target {
private:
  PathComponents Components;
  const Kind *K;
  Exactness::Values Exact;

public:
  Target(PathComponents Components,
         const Kind &K,
         Exactness::Values Exactness = Exactness::Exact) :
    Components(std::move(Components)), K(&K), Exact(Exactness) {
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(const Kind &K, Exactness::Values Exactness = Exactness::Exact) :
    K(&K), Exact(Exactness) {
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(PathComponent PathComponent,
         const Kind &K,
         Exactness::Values Exactness = Exactness::Exact) :
    Components({ std::move(PathComponent) }), K(&K), Exact(Exactness) {
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(std::string Name,
         const Kind &K,
         Exactness::Values Exactness = Exactness::Exact) :
    Components({ PathComponent(std::move(Name)) }), K(&K), Exact(Exactness) {

    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(std::initializer_list<std::string> Names,
         const Kind &K,
         Exactness::Values Exactness = Exactness::Exact) :
    K(&K), Exact(Exactness) {
    for (auto Name : Names)
      Components.emplace_back(std::move(Name));
    revng_assert(this->Components.size() == getKind().depth());
  }

  Target(llvm::ArrayRef<llvm::StringRef> Names,
         const Kind &K,
         Exactness::Values Exactness = Exactness::Exact) :
    K(&K), Exact(Exactness) {
    for (auto Name : Names)
      Components.emplace_back(Name.str());
    revng_assert(this->Components.size() == getKind().depth());
  }

public:
  bool operator<(const Target &Other) const {
    auto Self = std::tie(Components, K, Exact);
    auto OtherSelf = std::tie(Other.Components, Other.K, Other.Exact);
    return Self < OtherSelf;
  }

  int operator<=>(const Target &Other) const;

  bool operator==(const Target &Other) const { return (*this <=> Other) == 0; }

public:
  const Kind &getKind() const { return *K; }
  Exactness::Values kindExactness() const { return Exact; }
  const PathComponents &getPathComponents() const { return Components; }
  bool satisfies(const Target &Target) const;

public:
  void setKind(const Kind &NewKind) { K = &NewKind; }
  void setExactness(Exactness::Values NewExactness) { Exact = NewExactness; }

  void addPathComponent() { Components.emplace_back(PathComponent::all()); }
  void dropPathComponent() { Components.pop_back(); }
  llvm::Error verify(const ContainerBase &Container) const {
    return K->verify(Container, *this);
  }

public:
  void expand(const Context &Ctx, TargetsList &Out) const {
    K->expandTarget(Ctx, *this, Out);
  }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const debug_function {
    indent(OS, Indentation);

    OS << (Exact == Exactness::DerivedFrom ? "derived from " : "exactly ")
       << K->name().str() << " with path: ";
    for (const auto &Entry : Components) {
      Entry.dump(OS);
      OS << "/";
    }
    OS << "\n";
  }

  std::string serialize() const;

  template<typename OStream, typename Range>
  static void dumpPathComponents(OStream &OS, Range R) debug_function {
    for (const auto &Entry : R) {
      Entry.dump(OS);
      OS << "/";
    }
  }

  void dump() const debug_function { dump(dbg); }
};

class KindsRegistry;

llvm::Expected<Target>
parseTarget(llvm::StringRef AsString, const KindsRegistry &Dict);

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
  TargetsList(List C) : Contained(std::move(C)) { removeDuplicates(); }

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
    removeDuplicates();
  }

  void merge(const TargetsList &Other);

  void push_back(const Target &Target) {
    Contained.push_back(Target);
    removeDuplicates();
  }

  template<typename... Args>
  auto erase(Args &&...A) {
    return Contained.erase(std::forward<Args>(A)...);
  }

public:
  void dump() const debug_function { dump(dbg); }

  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    for (const auto &Entry : Contained)
      Entry.dump(OS, Indentation);
  }

private:
  void removeDuplicates();
};

/// A map from container name to target list that is usually used to represents
/// the state of a step.
class ContainerToTargetsMap {
public:
  using Map = llvm::StringMap<TargetsList>;
  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;
  using value_type = Map::value_type;

private:
  Map Status;

public:
  iterator begin() { return Status.begin(); }
  iterator end() { return Status.end(); }
  const_iterator begin() const { return Status.begin(); }
  const_iterator end() const { return Status.end(); }

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
           const Kind &K,
           Exactness::Values Exactness = Exactness::Exact) {
    Status[Name].emplace_back(Names, K, Exactness);
  }

  void add(llvm::StringRef Name, Target Target) {
    Status[Name].emplace_back(std::move(Target));
  }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    indent(OS, Indentation);
    OS << "{\n";
    for (const auto &Container : Status) {
      indent(OS, Indentation + 1);
      OS << Container.first().str() << ":\n";
      Container.second.dump(OS, Indentation + 2);
    }
    indent(OS, Indentation);
    OS << "}\n";
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

llvm::Error parseTarget(ContainerToTargetsMap &CurrentStatus,
                        llvm::StringRef AsString,
                        const KindsRegistry &Dict);

void prettyPrintTarget(const Target &Target,
                       llvm::raw_ostream &OS,
                       size_t Indentation = 0);

void prettyPrintStatus(const ContainerToTargetsMap &Targets,
                       llvm::raw_ostream &OS,
                       size_t Indentation = 0);

} // namespace pipeline
