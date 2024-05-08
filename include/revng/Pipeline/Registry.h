#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace pipeline {

class Loader;
class KindsRegistry;

/// A registry is the general way of registering factories and kinds from down
/// stream libraries.
///
/// It is possible that a factory might require to look up something into the
/// context and thus this class used to that purpose.
///
/// For those kinds and containers and pipes that do not require to look inside
/// the loader look further into this file to fine more declarative classes.
class Registry {
protected:
  Registry() { getInstances().push_back(this); }

public:
  virtual ~Registry() = default;

private:
  static llvm::SmallVector<Registry *, 4> &getInstances() {
    static llvm::SmallVector<Registry *, 4> Instances;
    return Instances;
  }

public:
  static void registerAllContainersAndPipes(Loader &Loader) {
    for (const auto &Reg : getInstances())
      Reg->registerContainersAndPipes(Loader);
  }

  static void runAllInitializationRoutines() {
    for (const auto &Reg : getInstances())
      Reg->libraryInitialization();
  }

  static KindsRegistry registerAllKinds();

public:
  virtual void registerContainersAndPipes(Loader &Loader) = 0;
  virtual void registerKinds(KindsRegistry &KindDictionary) = 0;
  virtual void libraryInitialization() = 0;
};

} // namespace pipeline
