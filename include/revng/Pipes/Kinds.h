#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Kind.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::pipes {

inline pipeline::Rank RootRank("Root Rank");

inline pipeline::Rank FunctionsRank("Function Rank", RootRank);

inline pipeline::Kind Binary("Binary", &RootRank);

inline RootKind Root("Root", &RootRank);
inline IsolatedRootKind IsolatedRoot("IsolatedRoot", Root);

inline TaggedFunctionKind
  Isolated("Isolated", &FunctionsRank, FunctionTags::Lifted);

inline pipeline::Kind Object("Object", &RootRank);
inline pipeline::Kind Translated("Translated", &RootRank);

} // namespace revng::pipes
