#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::pipes {

inline pipeline::Rank RootRank("root");

inline pipeline::Rank FunctionsRank("function", RootRank);

inline pipeline::Kind Binary("Binary", &RootRank);

inline RootKind Root("Root", &RootRank);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", Root);

inline TaggedFunctionKind
  Isolated("Isolated", &FunctionsRank, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("ABIEnforced", &FunctionsRank, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("CSVsPromoted", &FunctionsRank, FunctionTags::CSVsPromoted);

inline pipeline::Kind Object("Object", &RootRank);
inline pipeline::Kind Translated("Translated", &RootRank);

inline pipeline::Kind
  FunctionAssemblyInternal("FunctionAssemblyInternal", &FunctionsRank);
inline pipeline::Kind
  FunctionAssemblyHTML("FunctionAssemblyHTML", &FunctionsRank);
inline pipeline::Kind
  FunctionControlFlowGraphSVG("FunctionControlFlowGraphSVG", &FunctionsRank);

} // namespace revng::pipes
