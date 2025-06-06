/// \file Runner.cpp
/// A target is a object that is associated to the content of a container to
/// describe it without knowing what is the real type of the content itself.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

using namespace pipeline;
using namespace std;
using namespace llvm;

bool TargetsList::contains(const Target &Target) const {
  return find(*this, Target) != end();
}

void TargetsList::merge(const TargetsList &Source) {
  copy(Source, back_inserter(Contained));
  llvm::sort(Contained);
  Contained.erase(unique(Contained.begin(), Contained.end()), Contained.end());
}

void ContainerToTargetsMap::erase(const ContainerToTargetsMap &Other) {
  for (const auto &Container : Other.Status) {
    const auto &ContainerName = Container.first();
    const auto &ContainerSymbols = Container.second;
    if (Status.find(ContainerName) == Status.end())
      continue;
    auto &ToRemoveFrom = Status[ContainerName];

    for (auto &Symbol : ContainerSymbols)
      ToRemoveFrom.erase(Symbol);
  }
}

void ContainerToTargetsMap::merge(const ContainerToTargetsMap &Other) {
  for (const auto &Container : Other.Status) {
    const auto &ContainerName = Container.first();
    const auto &ContainerSymbols = Container.second;
    auto &ToMergeIn = Status[ContainerName];

    ToMergeIn.merge(ContainerSymbols);
  }
}

// NOTE: this operator needs to be stable w.r.t. library load order and memory
// layout
int Target::operator<=>(const Target &Other) const {
  if (K->id() < Other.K->id())
    return -1;
  if (K->id() > Other.K->id())
    return 1;

  if (Components.size() != Other.Components.size()) {
    if (Components.size() > Other.Components.size())
      return -1;
    if (Components.size() < Other.Components.size())
      return 1;
  }

  for (const auto &[l, r] : zip(Components, Other.Components)) {
    if (l < r)
      return -1;
    if (l > r)
      return 1;
  }

  return 0;
}

std::string Target::path() const {
  return llvm::join(Components, "/");
}

std::string Target::toString() const {
  return path() + ":" + K->name().str();
}

llvm::Expected<Target> Target::deserialize(Context &Context,
                                           llvm::StringRef String) {
  if (String.contains('*'))
    return revng::createError("String cannot contain *");

  TargetsList Out;
  if (auto Error = parseTarget(Context,
                               String,
                               Context.getKindsRegistry(),
                               Out);
      Error) {
    return std::move(Error);
  }

  revng_assert(Out.size() == 1);
  return Out.front();
}

llvm::Error pipeline::parseTarget(const Context &Context,
                                  llvm::StringRef AsString,
                                  const KindsRegistry &Dict,
                                  TargetsList &Out) {

  size_t Pos = AsString.rfind(':');

  if (Pos == llvm::StringRef::npos) {
    auto *Message = "String '%s' was not in expected form <path:kind>";
    return revng::createError(Message, AsString.str().c_str());
  }

  llvm::StringRef Name(AsString.data(), Pos);
  llvm::StringRef KindName = AsString.drop_front(Pos + 1);
  llvm::SmallVector<llvm::StringRef, 4> Path;
  Name.split(Path, '/');

  auto It = llvm::find_if(Dict, [&](Kind &K) { return KindName == K.name(); });
  if (It == Dict.end())
    return revng::createError("No known Kind '%s' in dictionary",
                              KindName.str().c_str());

  if (AsString[0] == ':') {
    Out.push_back(Target({}, *It));
    return llvm::Error::success();
  }

  if (find(AsString, '*') != AsString.end()) {
    It->appendAllTargets(Context, Out);
    return llvm::Error::success();
  }

  Out.push_back(Target(std::move(Path), *It));
  return llvm::Error::success();
}

llvm::Error pipeline::parseTarget(const Context &Context,
                                  ContainerToTargetsMap &CurrentStatus,
                                  llvm::StringRef AsString,
                                  const KindsRegistry &Dict) {
  if (AsString.empty())
    return llvm::Error::success();

  llvm::SmallVector<llvm::StringRef, 2> Parts;
  AsString.split(Parts, '/', 1);

  if (Parts.size() != 2) {
    auto *Text = "string '%s' was not in expected form <ContainerName/Target>";
    return revng::createError(Text, AsString.str().c_str());
  }

  return parseTarget(Context, Parts[1], Dict, CurrentStatus[Parts[0]]);
}
