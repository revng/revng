#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Pipes/TypeKind.h"

namespace revng::kinds {

template<typename... T>
inline std::tuple<const T &...> fat(const T &...Refs) {
  return std::forward_as_tuple(Refs...);
}

inline pipeline::SingleElementKind Binary("binary", ranks::Binary, {}, {});
inline pipeline::SingleElementKind HexDump("hex-dump", ranks::Binary, {}, {});

inline RootKind Root("root", ranks::Binary);
inline IsolatedRootKind IsolatedRoot("isolated-root", Root, ranks::Binary);

inline TaggedFunctionKind
  Isolated("isolated", ranks::Function, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("abi-enforced", ranks::Function, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("csvs-promoted", ranks::Function, FunctionTags::CSVsPromoted);

inline pipeline::SingleElementKind Object("object", ranks::Binary, {}, {});
inline pipeline::SingleElementKind
  Translated("translated", ranks::Binary, {}, {});

inline FunctionKind CFG("cfg", ranks::Function, {}, {});

inline FunctionKind FunctionAssemblyInternal("function-assembly-internal",
                                             ranks::Function,
                                             {},
                                             {});

inline FunctionKind FunctionAssemblyPTML("function-assembly-ptml",
                                         ranks::Function,
                                         fat(ranks::Function,
                                             ranks::BasicBlock,
                                             ranks::Instruction),
                                         {});

inline auto FunctionControlFlowGraphSVGName = "function-control-flow-graph-svg";
inline FunctionKind FunctionControlFlowGraphSVG(FunctionControlFlowGraphSVGName,
                                                ranks::Function,
                                                {},
                                                {});

inline pipeline::SingleElementKind
  BinaryCrossRelations("binary-cross-relations", ranks::Binary, {}, {});
inline pipeline::SingleElementKind
  CallGraphSVG("call-graph-svg", ranks::Binary, {}, {});
inline FunctionKind
  CallGraphSliceSVG("call-graph-slice-svg", ranks::Function, {}, {});

inline constexpr auto BinaryCrossRelationsRole = "cross-relations";

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

inline FunctionKind CliftFunction("clift-module", ranks::Function, {}, {});

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
