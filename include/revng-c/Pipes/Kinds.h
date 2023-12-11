#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/TaggedFunctionKind.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/FunctionTags.h"

namespace revng::kinds {

inline TaggedFunctionKind
  LiftingArtifactsRemoved("LiftingArtifactsRemoved",
                          ranks::Function,
                          FunctionTags::LiftingArtifactsRemoved);

inline TaggedFunctionKind
  StackPointerPromoted("StackPointerPromoted",
                       ranks::Function,
                       FunctionTags::StackPointerPromoted);

inline TaggedFunctionKind
  StackAccessesSegregated("StackAccessesSegregated",
                          ranks::Function,
                          FunctionTags::StackAccessesSegregated);

extern FunctionKind DecompiledToYAML;
inline pipeline::SingleElementKind ModelHeader("ModelHeader",
                                               Binary,
                                               ranks::Binary,
                                               fat(ranks::Type,
                                                   ranks::StructField,
                                                   ranks::UnionField,
                                                   ranks::EnumEntry,
                                                   ranks::DynamicFunction,
                                                   ranks::Segment,
                                                   ranks::ArtificialStruct),
                                               { &DecompiledToYAML });

inline FunctionKind DecompiledToYAML("DecompiledToYAML",
                                     ModelHeader,
                                     ranks::Function,
                                     fat(ranks::Function),
                                     { &ModelHeader });

inline pipeline::SingleElementKind
  HelpersHeader("HelpersHeader", Binary, ranks::Binary, {}, {});

inline pipeline::SingleElementKind
  MLIRLLVMModule("MLIRLLVMModule", Binary, ranks::Binary, {}, {});

inline pipeline::SingleElementKind DecompiledToC("DecompiledToC",
                                                 Binary,
                                                 ranks::Binary,
                                                 fat(ranks::Function),
                                                 { &ModelHeader });

} // namespace revng::kinds
