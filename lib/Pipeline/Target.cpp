/// \file Runner.cpp
/// \brief a target is a object that is associated to the content of a container
/// to describe it without knowing what is the real type of the content itself.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;
using namespace std;
using namespace llvm;

void TargetsList::removeDuplicates() {
  sort(Contained);
  const auto IsSuperSet = [](const Target &L, const Target &R) {
    return L.satisfies(R);
  };
  Contained.erase(unique(begin(), end(), IsSuperSet), end());
}

bool TargetsList::contains(const Target &Target) const {
  const auto IsCompatible = [&Target](const pipeline::Target &ToCheck) {
    return ToCheck.satisfies(Target);
  };
  return find_if(*this, IsCompatible) != end();
}

void TargetsList::merge(const TargetsList &Source) {
  copy(Source, back_inserter(Contained));
  removeDuplicates();
}

void ContainerToTargetsMap::merge(const ContainerToTargetsMap &Other) {
  for (const auto &Container : Other.Status) {
    const auto &ContainerName = Container.first();
    const auto &ContainerSymbols = Container.second;
    auto &ToMergeIn = Status[ContainerName];

    ToMergeIn.merge(ContainerSymbols);
  }
}

int Target::operator<=>(const Target &Other) const {
  if (K > Other.K)
    return -1;
  if (K < Other.K)
    return 1;

  if (Exact > Other.Exact)
    return -1;
  if (Exact < Other.Exact)
    return 1;

  if (Components.size() != Other.Components.size()) {
    if (Components.size() > Other.Components.size())
      return -1;
    if (Components.size() < Other.Components.size())
      return 1;
  }

  for (const auto &[l, r] : zip(Components, Other.Components)) {
    auto Val = l <=> r;
    if (Val != 0)
      return Val;
  }

  return 0;
}

bool Target::satisfies(const Target &Other) const {
  // if they require a exact kind match and we are different return false
  if (Other.kindExactness() == Exactness::Exact
      and &Other.getKind() != &getKind())
    return false;

  // if they require a derived match and they are not our ancestor return false
  if (Other.kindExactness() == Exactness::DerivedFrom
      and not Other.getKind().ancestorOf(getKind()))
    return false;

  revng_assert(getPathComponents().size() == Other.getPathComponents().size());

  // for all path components
  for (size_t I = 0; I < getPathComponents().size(); I++) {
    // if describe all it does not matter what they are
    if (getPathComponents()[I].isAll())
      continue;

    // otherwise we must be identical else we return
    if (getPathComponents()[I] != Other.getPathComponents()[I])
      return false;
  }
  return true;
}

std::string Target::serialize() const {
  std::string ToReturn;

  if (Components.size() == 0) {
    return ":" + K->name().str();
  }

  for (size_t I = 0; I < Components.size() - 1; I++)
    ToReturn += Components[I].toString() + "/";

  ToReturn += Components.back().toString();
  ToReturn += ":";
  ToReturn += K->name();

  return ToReturn;
}

llvm::Expected<Target>
pipeline::parseTarget(llvm::StringRef AsString, const KindsRegistry &Dict) {

  llvm::SmallVector<llvm::StringRef, 2> Parts;
  AsString.split(Parts, ':', 2);

  if (Parts.size() != 2) {
    auto *Message = "string %s was not in expected form <path:kind>";
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   Message,
                                   AsString.str().c_str());
  }

  llvm::SmallVector<llvm::StringRef, 4> Path;
  Parts[0].split(Path, '/');

  auto It = llvm::find_if(Dict,
                          [&Parts](Kind &K) { return Parts[1] == K.name(); });
  if (It == Dict.end())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known Kind %s in dictionary",
                                   Parts[1].str().c_str());

  if (AsString[0] == ':')
    return Target({}, *It);

  return Target(std::move(Path), *It);
}

llvm::Error pipeline::parseTarget(ContainerToTargetsMap &CurrentStatus,
                                  llvm::StringRef AsString,
                                  const KindsRegistry &Dict) {
  llvm::SmallVector<llvm::StringRef, 2> Parts;
  AsString.split(Parts, ':', 1);

  if (Parts.size() != 2) {
    auto *Message = "string %s was not in expected form <ContainerName:Target>";
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   Message,
                                   AsString.str().c_str());
  }

  auto MaybeTarget = parseTarget(Parts[1], Dict);
  if (not MaybeTarget)
    return MaybeTarget.takeError();

  CurrentStatus.add(Parts[0], std::move(*MaybeTarget));
  return llvm::Error::success();
}

void pipeline::prettyPrintTarget(const Target &Target,
                                 llvm::raw_ostream &OS,
                                 size_t Indentation) {
  OS.indent(Indentation);

  OS.changeColor(llvm::raw_ostream::Colors::CYAN);
  OS << Target.getKind().name();

  OS << " ";

  OS.changeColor(llvm::raw_ostream::Colors::YELLOW);
  for (const auto &PathComponent : Target.getPathComponents())
    OS << (PathComponent.isAll() ? "*" : PathComponent.getName()) << "/";

  OS << "\n";
}

void pipeline::prettyPrintStatus(const ContainerToTargetsMap &Targets,
                                 llvm::raw_ostream &OS,
                                 size_t Indentation) {
  for (const auto &Pair : Targets) {
    const auto &Name = Pair.first();
    const auto &List = Pair.second;
    if (List.empty())
      continue;

    OS.changeColor(llvm::raw_ostream::Colors::BLUE);
    OS.indent(Indentation);
    OS << Name << "\n";

    for (const auto &Target : List)
      prettyPrintTarget(Target, OS, Indentation + 1);
  }
}
