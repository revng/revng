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

inline pipeline::Kind Binary("Binary", &ranks::Root);

inline RootKind Root("Root", &ranks::Root);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", &ranks::Root);

inline TaggedFunctionKind
  Isolated("Isolated", &ranks::Function, FunctionTags::Isolated);
inline TaggedFunctionKind
  ABIEnforced("ABIEnforced", &ranks::Function, FunctionTags::ABIEnforced);
inline TaggedFunctionKind
  CSVsPromoted("CSVsPromoted", &ranks::Function, FunctionTags::CSVsPromoted);

inline pipeline::Kind Object("Object", &ranks::Root);
inline pipeline::Kind Translated("Translated", &ranks::Root);

inline FunctionKind
  FunctionAssemblyInternal("FunctionAssemblyInternal", &ranks::Function);
inline FunctionKind
  FunctionAssemblyPTML("FunctionAssemblyPTML", &ranks::Function);
inline FunctionKind
  FunctionControlFlowGraphSVG("FunctionControlFlowGraphSVG", &ranks::Function);

} // namespace revng::pipes
