/// \file Runner.cpp
/// \brief a target is a object that is associated to the content of a container
/// to describe it without knowing what is the real type of the content itself.

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

void ContainerToTargetsMap::merge(const ContainerToTargetsMap &Other) {
  for (const auto &Container : Other.Status) {
    const auto &ContainerName = Container.first();
    const auto &ContainerSymbols = Container.second;
    auto &ToMergeIn = Status[ContainerName];

    ToMergeIn.merge(ContainerSymbols);
  }
}

int Target::operator<=>(const Target &Other) const {
  if (K < Other.K)
    return -1;
  if (K > Other.K)
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

std::string Target::serialize() const {
  std::string ToReturn;

  if (Components.size() == 0) {
    return ":" + K->name().str();
  }

  for (size_t I = 0; I < Components.size() - 1; I++)
    ToReturn += Components[I] + "/";

  ToReturn += Components.back();
  ToReturn += ":";
  ToReturn += K->name();

  return ToReturn;
}

llvm::Expected<Target> Target::deserialize(Context &Ctx,
                                           const KindsRegistry &Dict,
                                           llvm::StringRef String) {
  if (String.contains('*'))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "String cannot contain *");

  TargetsList Out;
  if (auto Error = parseTarget(Ctx, String, Dict, Out); Error) {
    return std::move(Error);
  }

  revng_assert(Out.size() == 1);
  return Out.front();
}

llvm::Error pipeline::parseTarget(const Context &Ctx,
                                  llvm::StringRef AsString,
                                  const KindsRegistry &Dict,
                                  TargetsList &Out) {

  llvm::SmallVector<llvm::StringRef, 2> Parts;
  AsString.split(Parts, ':', 2);

  if (Parts.size() != 2) {
    auto *Message = "string '%s' was not in expected form <path:kind>";
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
                                   "No known Kind '%s' in dictionary",
                                   Parts[1].str().c_str());

  if (AsString[0] == ':') {
    Out.push_back(Target({}, *It));
    return llvm::Error::success();
  }

  if (find(AsString, '*') != AsString.end()) {
    It->appendAllTargets(Ctx, Out);
    return llvm::Error::success();
  }

  Out.push_back(Target(std::move(Path), *It));
  return llvm::Error::success();
}

llvm::Error pipeline::parseTarget(const Context &Ctx,
                                  ContainerToTargetsMap &CurrentStatus,
                                  llvm::StringRef AsString,
                                  const KindsRegistry &Dict) {
  if (AsString.empty())
    return llvm::Error::success();

  llvm::SmallVector<llvm::StringRef, 2> Parts;
  AsString.split(Parts, '/', 1);

  if (Parts.size() != 2) {
    auto *Text = "string '%s' was not in expected form <ContainerName/Target>";
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   Text,
                                   AsString.str().c_str());
  }

  return parseTarget(Ctx, Parts[1], Dict, CurrentStatus[Parts[0]]);
}
