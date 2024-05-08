// \file RegisterKind.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/RegisterKind.h"

namespace pipeline {

void RegisterKind::registerKinds(KindsRegistry &KindDictionary) {
  KindDictionary.registerKind(*K);
}

} // namespace pipeline
