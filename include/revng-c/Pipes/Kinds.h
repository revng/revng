#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/TaggedFunctionKind.h"

#include "revng-c/Support/FunctionTags.h"

namespace revng::pipes {

inline TaggedFunctionKind
  LiftingArtifactsRemoved("LiftingArtifactsRemoved",
                          &FunctionsRank,
                          FunctionTags::LiftingArtifactsRemoved);

} // end namespace revng::pipes
