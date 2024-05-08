#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Registry.h"

namespace pipeline {

class Kind;

/// Instantiate a global object of this class for each kind you wish to
/// register
class RegisterKind : Registry {
private:
  Kind *K;

public:
  RegisterKind(Kind &K) : K(&K) {}

  ~RegisterKind() override = default;

public:
  void registerContainersAndPipes(Loader &) override {}
  void registerKinds(KindsRegistry &KindDictionary) override;
  void libraryInitialization() override {}
};

} // namespace pipeline
