#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::kinds {

template<typename... T>
inline std::tuple<const T &...> fat(const T &...Refs) {
  return std::forward_as_tuple(Refs...);
}

inline pipeline::SingleElementKind Binary("Binary", ranks::Binary, {}, {});
inline pipeline::SingleElementKind HexDump("HexDump", ranks::Binary, {}, {});

inline RootKind Root("Root", ranks::Binary);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", Root, ranks::Binary);

inline TaggedFunctionKind
  Isolated("Isolated", ranks::Function, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("ABIEnforced", ranks::Function, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("CSVsPromoted", ranks::Function, FunctionTags::CSVsPromoted);

inline pipeline::SingleElementKind Object("Object", ranks::Binary, {}, {});
inline pipeline::SingleElementKind
  Translated("Translated", ranks::Binary, {}, {});

inline FunctionKind
  FunctionAssemblyInternal("FunctionAssemblyInternal", ranks::Function, {}, {});

inline FunctionKind FunctionAssemblyPTML("FunctionAssemblyPTML",
                                         ranks::Function,
                                         fat(ranks::Function,
                                             ranks::BasicBlock,
                                             ranks::Instruction),
                                         {});

inline FunctionKind FunctionControlFlowGraphSVG("FunctionControlFlowGraphSVG",
                                                ranks::Function,
                                                {},
                                                {});

inline pipeline::SingleElementKind
  BinaryCrossRelations("BinaryCrossRelations", ranks::Binary, {}, {});
inline pipeline::SingleElementKind
  CallGraphSVG("CallGraphSVG", ranks::Binary, {}, {});
inline FunctionKind
  CallGraphSliceSVG("CallGraphSliceSVG", ranks::Function, {}, {});

inline constexpr auto BinaryCrossRelationsRole = "CrossRelations";

} // namespace revng::kinds
