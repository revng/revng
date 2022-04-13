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

inline TaggedFunctionKind
  StackPointerPromoted("StackPointerPromoted",
                       &FunctionsRank,
                       FunctionTags::StackPointerPromoted);

inline TaggedFunctionKind
  StackAccessesSegregated("StackAccessesSegregated",
                          &FunctionsRank,
                          FunctionTags::StackAccessesSegregated);

inline pipeline::Kind ModelHeader("ModelHeader", Binary, &RootRank);

} // end namespace revng::pipes
