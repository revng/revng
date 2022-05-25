#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <initializer_list>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Global.h"
#include "revng/Pipeline/GlobalsMap.h"
#include "revng/Pipeline/KindsRegistry.h"

namespace pipeline {

/// A class that contains every object that has a lifetime longer than a
/// pipeline.
///
/// This includes the kinds and the NamedGlobals that will be available in that
/// context.
class Context {
private:
  GlobalsMap Globals;
  llvm::StringMap<std::any> Contexts;
  KindsRegistry TheKindRegistry;

private:
  explicit Context(KindsRegistry Registry) :
    TheKindRegistry(std::move(Registry)) {}

public:
  Context();

public:
  static Context fromRegistry(KindsRegistry Registry) {
    return Context(std::move(Registry));
  }

public:
  const KindsRegistry &getKindsRegistry() const { return TheKindRegistry; }

  template<typename T>
  llvm::Expected<T *> getGlobal(llvm::StringRef Name) const {
    return Globals.get<T>(Name);
  }

  llvm::Error
  serializeGlobal(llvm::StringRef GlobalName, llvm::raw_ostream &OS) const {
    return Globals.serialize(GlobalName, OS);
  }

  llvm::Error deserializeGlobal(llvm::StringRef GlobalName,
                                const llvm::MemoryBuffer &Buffer) {
    return Globals.deserialize(GlobalName, Buffer);
  }

  template<typename T, typename... ArgsT>
  void addGlobal(llvm::StringRef Name, ArgsT &&...Args) {
    Globals.emplace<T>(Name, std::forward<T>(Args)...);
  }

  template<typename T>
  void addExternalContext(llvm::StringRef Name, T &ToAdd) {
    Contexts.try_emplace(Name, std::make_any<T *>(&ToAdd));
  }

  template<typename T>
  llvm::Expected<T *> getExternalContext(llvm::StringRef Name) const {
    auto Iter = Contexts.find(Name);
    if (Iter == Contexts.end()) {
      auto *Message = "pipeline loader context did not contained object %s";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    auto *Casted = std::any_cast<T *>(Iter->second);
    if (Casted == nullptr) {
      auto *Message = "pipeline loader was requested to cast %s to the wrong "
                      "type";
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Message,
                                     Name.str().c_str());
    }

    return Casted;
  }

  const GlobalsMap &getGlobals() const { return Globals; }

public:
  llvm::Error storeToDisk(llvm::StringRef Path) const {
    return Globals.storeToDisk(Path);
  }
  llvm::Error loadFromDisk(llvm::StringRef Path) {
    return Globals.loadFromDisk(Path);
  }
};

} // namespace pipeline
