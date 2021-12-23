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

#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/SavableObject.h"

namespace Pipeline {

/// a simple pair that holds a savable object a name meant to describe a object
/// which lifetime must be than the context.
class NamedGlobalReference {
private:
  llvm::StringRef Name;
  SavableObjectBase *Obj;

public:
  NamedGlobalReference(llvm::StringRef Name, SavableObjectBase &Obj) :
    Name(Name), Obj(&Obj) {}

  llvm::StringRef name() const { return Name; }

  SavableObjectBase &value() const { return *Obj; }
};

/// A class that contains every object that has a lifetime longer
/// than a pipeline.
///
/// This includes the kinds and the NamedGlobals that will be available in that
/// context
class Context {
private:
  llvm::StringMap<SavableObjectBase *> Globals;
  KindsRegistry TheKindRegistry;

  explicit Context(KindsRegistry registry) :
    TheKindRegistry(std::move(registry)) {}
  Context(llvm::ArrayRef<NamedGlobalReference> Globals, KindsRegistry registry);

public:
  Context();
  Context(llvm::ArrayRef<NamedGlobalReference> Globals);

  static Context fromRegistry(llvm::ArrayRef<NamedGlobalReference> Globals,
                              KindsRegistry registry) {
    return Context(Globals, std::move(registry));
  }

  static Context fromRegistry(KindsRegistry registry) {
    return Context(std::move(registry));
  }

  template<typename T>
  llvm::Expected<T *> getGlobal(llvm::StringRef Name) const {
    auto Iter = Globals.find(Name);
    if (Iter == Globals.end())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "pipeline loader context did not "
                                     "contained object %s",
                                     Name.str().c_str());
    auto *Casted = llvm::dyn_cast<T>(Iter->second);
    if (Casted == nullptr)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "pipeline loader was requested to cast %s "
                                     "to the wrong type",
                                     Name.str().c_str());

    return Casted;
  }

  const KindsRegistry &getKindsRegistry() const { return TheKindRegistry; }

  llvm::Error storeToDisk(llvm::StringRef Path) const;
  llvm::Error loadFromDisk(llvm::StringRef Path);

  template<typename... T>
  static Context createFromGlobals(T &&...Args) {
    using namespace std;
    using namespace llvm;
    SmallVector<NamedGlobalReference, 4> Globals({ forward<T>(Args)... });
    return Context(Globals);
  }
};

} // namespace Pipeline
