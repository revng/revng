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

namespace Pipeline {

enum class Exactness { Exact, DerivedFrom };

class TargetsList;

/// A target is a triple of Kind, PathComponents List, and Exactness used to
/// enumerate and transform the contents of a container.
///
/// The kind is used to tell apart objects that are conceptually different or
/// have a relationship of containment or extension The PathComponent list is
/// used to tell apart objects belonging to the same Kind The Exactness express
/// that a requirements can be satisfied by a kind or its extension
class Target {
private:
  PathComponents Components;
  const Kind *K;
  Exactness Exact;

public:
  Target(PathComponents Components,
         const Kind &K,
         Exactness Exactness = Exactness::Exact) :
    Components(std::move(Components)), K(&K), Exact(Exactness) {
    revng_assert(this->Components.size() == getKind().depth() + 1);
  }

  Target(PathComponent PathComponent,
         const Kind &K,
         Exactness Exactness = Exactness::Exact) :
    Components({ std::move(PathComponent) }), K(&K), Exact(Exactness) {
    revng_assert(this->Components.size() == getKind().depth() + 1);
  }

  Target(std::string Name,
         const Kind &K,
         Exactness Exactness = Exactness::Exact) :
    Components({ PathComponent(std::move(Name)) }), K(&K), Exact(Exactness) {

    revng_assert(this->Components.size() == getKind().depth() + 1);
  }

  Target(std::initializer_list<std::string> Names,
         const Kind &K,
         Exactness Exactness = Exactness::Exact) :
    K(&K), Exact(Exactness) {
    for (auto Name : Names)
      Components.emplace_back(std::move(Name));
    revng_assert(this->Components.size() == getKind().depth() + 1);
  }

  Target(llvm::ArrayRef<llvm::StringRef> Names,
         const Kind &K,
         Exactness Exactness = Exactness::Exact) :
    K(&K), Exact(Exactness) {
    for (auto Name : Names)
      Components.emplace_back(Name.str());
    revng_assert(this->Components.size() == getKind().depth() + 1);
  }

  bool operator<(const Target &Other) const {
    auto Self = std::tie(Components, K, Exact);
    auto OtherSelf = std::tie(Other.Components, Other.K, Other.Exact);
    return Self < OtherSelf;
  }

  const Kind &getKind() const { return *K; }

  Exactness kindExactness() const { return Exact; }
  void setKind(const Kind &NewKind) { K = &NewKind; }
  void setExactness(Exactness NewExactness) { Exact = NewExactness; }

  const PathComponents &getPathComponents() const { return Components; }

  void addPathComponent() { Components.emplace_back(); }
  void dropPathComponent() { Components.pop_back(); }

  template<typename OStream>
  void dump(OStream &OS, size_t Indents = 0) const debug_function {
    indent(OS, Indents);

    OS << (Exact == Exactness::DerivedFrom ? "derived from " : "exactly ")
       << K->name().str() << " With path: ";
    for (const auto &Entry : Components) {
      Entry.dump(OS);
      OS << "/";
    }
    OS << "\n";
  }

  template<typename OStream, typename Range>
  static void dumpPathComponents(OStream &OS, Range R) debug_function {
    for (const auto &Entry : R) {
      Entry.dump(OS);
      OS << "/";
    }
  }

  void expand(const Context &Ctx, TargetsList &Out) const {
    K->expandTarget(Ctx, *this, Out);
  }

  std::string serialize() const;

  void dump() const debug_function { dump(dbg); }

  int operator<=>(const Target &Other) const;

  bool operator==(const Target &Other) const { return (*this <=> Other) == 0; }

  bool satisfies(const Target &Target) const;
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

  iterator begin() { return Contained.begin(); }

  iterator end() { return Contained.end(); }

  const_iterator begin() const { return Contained.begin(); }

  const_iterator end() const { return Contained.end(); }

  size_t size() const { return Contained.size(); }

  template<typename... Args>
  void emplace_back(Args &&...args) {
    Contained.emplace_back(std::forward<Args>(args)...);
    removeDuplicates();
  }

  void dump() const debug_function { dump(dbg); }

  template<typename OStream>
  void dump(OStream &OS, size_t Indents = 0) const {
    for (const auto &Entry : Contained)
      Entry.dump(OS, Indents);
  }

  void merge(const TargetsList &Other);

  void push_back(const Target &Target) {
    Contained.push_back(Target);
    removeDuplicates();
  }

  const Target &operator[](size_t Index) const { return Contained[Index]; }

  Target &operator[](size_t Index) { return Contained[Index]; }

  template<typename... Args>
  auto erase(Args &&...args) {
    return Contained.erase(std::forward<Args>(args)...);
  }

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

  bool empty() const { return Contained.empty(); }

private:
  void removeDuplicates();
};

/// a map from container name to target list that is usually used to represents
/// the state of a step.
class ContainerToTargetsMap {
public:
  using Map = llvm::StringMap<TargetsList>;
  using iterator = Map::iterator;
  using const_iterator = Map::const_iterator;
  using value_type = Map::value_type;

private:
  Map Status;

  size_t targetsCount() const {
    size_t Size = 0;
    for (const auto &Container : Status)
      Size += Container.second.size();
    return Size;
  }

public:
  TargetsList &operator[](llvm::StringRef ContainerName) {
    return Status[ContainerName];
  }

  TargetsList &at(llvm::StringRef Name) {
    revng_assert(Status.find(Name) != Status.end());
    return Status.find(Name)->getValue();
  }

  const TargetsList &at(llvm::StringRef ContainerName) const {
    revng_assert(Status.find(ContainerName) != Status.end());
    return Status.find(ContainerName)->second;
  }

  void merge(const ContainerToTargetsMap &Other);

  void add(llvm::StringRef Name,
           std::initializer_list<std::string> Names,
           const Kind &K,
           Exactness Exactness = Exactness::Exact) {
    Status[Name].emplace_back(Names, K, Exactness);
  }

  void add(llvm::StringRef Name, Target Target) {
    Status[Name].emplace_back(std::move(Target));
  }

public:
  iterator begin() { return Status.begin(); }
  iterator end() { return Status.end(); }
  const_iterator begin() const { return Status.begin(); }
  const_iterator end() const { return Status.end(); }

  bool empty() const { return targetsCount() == 0; }

  bool contains(llvm::StringRef ContainerName) const {
    return Status.find(ContainerName) != Status.end();
  }

  iterator find(llvm::StringRef ContainerName) {
    return Status.find(ContainerName);
  }

  const_iterator find(llvm::StringRef ContainerName) const {
    return Status.find(ContainerName);
  }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indents = 0) const {
    indent(OS, Indents);
    OS << "{\n";
    for (const auto &Container : Status) {
      indent(OS, Indents + 1);
      OS << Container.first().str() << ":\n";
      Container.second.dump(OS, Indents + 2);
    }
    indent(OS, Indents);
    OS << "}\n";
  }
  void dump() const debug_function { dump(dbg); }
};

llvm::Error parseTarget(ContainerToTargetsMap &CurrentStatus,
                        llvm::StringRef AsString,
                        const KindsRegistry &Dict);

void prettyPrintTarget(const Target &Target,
                       llvm::raw_ostream &OS,
                       size_t Indents = 0);

void prettyPrintStatus(const ContainerToTargetsMap &Targets,
                       llvm::raw_ostream &OS,
                       size_t Indents = 0);
} // namespace Pipeline
