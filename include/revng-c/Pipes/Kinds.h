#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/TaggedFunctionKind.h"

#include "revng-c/Support/FunctionTags.h"

namespace revng::kinds {

inline TaggedFunctionKind
  LiftingArtifactsRemoved("LiftingArtifactsRemoved",
                          &ranks::Function,
                          FunctionTags::LiftingArtifactsRemoved);

inline TaggedFunctionKind
  StackPointerPromoted("StackPointerPromoted",
                       &ranks::Function,
                       FunctionTags::StackPointerPromoted);

inline TaggedFunctionKind
  StackAccessesSegregated("StackAccessesSegregated",
                          &ranks::Function,
                          FunctionTags::StackAccessesSegregated);

inline FunctionKind DecompiledToYAML("DecompiledToYAML", &ranks::Function);

inline pipeline::SingleElementKind
  ModelHeader("ModelHeader", Binary, &ranks::Binary);

inline pipeline::SingleElementKind
  HelpersHeader("HelpersHeader", Binary, &ranks::Binary);

inline pipeline::SingleElementKind
  DecompiledToC("DecompiledToC", Binary, &ranks::Binary);

} // namespace revng::kinds
