#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Pipes/TypeKind.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Pipes/Ranks.h"

namespace revng::kinds {

inline TaggedFunctionKind
  LiftingArtifactsRemoved("lifting-artifacts-removed",
                          ranks::Function,
                          FunctionTags::LiftingArtifactsRemoved);

inline TaggedFunctionKind
  StackPointerPromoted("stack-pointer-promoted",
                       ranks::Function,
                       FunctionTags::StackPointerPromoted);

inline TaggedFunctionKind
  StackAccessesSegregated("stack-accesses-segregated",
                          ranks::Function,
                          FunctionTags::StackAccessesSegregated);

extern FunctionKind Decompiled;
inline pipeline::SingleElementKind ModelHeader("model-header",
                                               Binary,
                                               ranks::Binary,
                                               fat(ranks::TypeDefinition,
                                                   ranks::StructField,
                                                   ranks::UnionField,
                                                   ranks::EnumEntry,
                                                   ranks::DynamicFunction,
                                                   ranks::Segment,
                                                   ranks::ArtificialStruct),
                                               { &Decompiled });

inline FunctionKind Decompiled("decompiled",
                               ModelHeader,
                               ranks::Function,
                               fat(ranks::Function),
                               { &ModelHeader });

inline TypeKind ModelTypeDefinition("model-type-definition",
                                    ModelHeader,
                                    ranks::TypeDefinition,
                                    {},
                                    {});

inline pipeline::SingleElementKind
  HelpersHeader("helpers-header", Binary, ranks::Binary, {}, {});

inline FunctionKind MLIRFunctionKind("mlir-module", ranks::Function, {}, {});

inline pipeline::SingleElementKind DecompiledToC("decompiled-to-c",
                                                 Binary,
                                                 ranks::Binary,
                                                 fat(ranks::Function),
                                                 { &ModelHeader });

inline pipeline::SingleElementKind
  RecompilableArchive("recompilable-archive",
                      Binary,
                      ranks::Binary,
                      fat(ranks::Function,
                          ranks::TypeDefinition,
                          ranks::StructField,
                          ranks::UnionField,
                          ranks::EnumEntry,
                          ranks::DynamicFunction,
                          ranks::Segment,
                          ranks::ArtificialStruct),
                      {});

} // namespace revng::kinds
