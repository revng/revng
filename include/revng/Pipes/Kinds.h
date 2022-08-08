#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::pipes {

inline pipeline::Kind Binary("Binary", &ranks::Binary);

inline RootKind Root("Root", &ranks::Binary);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", Root);

inline TaggedFunctionKind
  Isolated("Isolated", &ranks::Function, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("ABIEnforced", &ranks::Function, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("CSVsPromoted", &ranks::Function, FunctionTags::CSVsPromoted);

inline pipeline::Kind Object("Object", &ranks::Binary);
inline pipeline::Kind Translated("Translated", &ranks::Binary);

inline FunctionKind
  FunctionAssemblyInternal("FunctionAssemblyInternal", &ranks::Function);
inline FunctionKind
  FunctionAssemblyPTML("FunctionAssemblyPTML", &ranks::Function);
inline FunctionKind
  FunctionControlFlowGraphSVG("FunctionControlFlowGraphSVG", &ranks::Function);

inline pipeline::Kind
  BinaryCrossRelations("BinaryCrossRelations", &ranks::Binary);

inline constexpr auto BinaryCrossRelationsRole = "CrossRelations";

} // namespace revng::pipes
