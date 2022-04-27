/// \file Context.cpp
/// \brief The pipeline context the place where all objects used by more that
/// one pipeline or container are stored.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"

using namespace pipeline;

Context::Context() : TheKindRegistry(Registry::registerAllKinds()) {
}
