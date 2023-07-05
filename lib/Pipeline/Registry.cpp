// \file Registry.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/Registry.h"

namespace pipeline {

KindsRegistry Registry::registerAllKinds() {
  KindsRegistry Registry;
  for (const auto &Reg : getInstances())
    Reg->registerKinds(Registry);

  Kind::init();
  Rank::init();
  return Registry;
}

} // namespace pipeline
