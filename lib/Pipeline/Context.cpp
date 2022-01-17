/// \file Context.cpp
/// \brief The pipeline context the place where all objects used by more that
/// one pipeline or container are stored.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Registry.h"

using namespace pipeline;

llvm::Error Context::storeToDisk(llvm::StringRef Path) const {
  for (const auto &Global : Globals)
    if (auto E = Global.second->storeToDisk(Path.str() + "/"
                                            + Global.first().str());
        !!E)
      return E;
  return llvm::Error::success();
}

llvm::Error Context::loadFromDisk(llvm::StringRef Path) {
  for (const auto &Global : Globals)
    if (auto E = Global.second->loadFromDisk(Path.str() + "/"
                                             + Global.first().str());
        !!E)
      return E;
  return llvm::Error::success();
}

Context::Context() : TheKindRegistry(Registry::registerAllKinds()) {
}

Context::Context(llvm::ArrayRef<NamedGlobalReference> Globals,
                 KindsRegistry Registry) :
  TheKindRegistry(std::move(Registry)) {
  for (auto &Global : Globals)
    this->Globals.try_emplace(Global.name(), &Global.value());
}

Context::Context(llvm::ArrayRef<NamedGlobalReference> Globals) :
  TheKindRegistry(Registry::registerAllKinds()) {
  for (auto &Global : Globals)
    this->Globals.try_emplace(Global.name(), &Global.value());
}
