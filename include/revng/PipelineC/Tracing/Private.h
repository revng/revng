#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/raw_ostream.h"

namespace revng::tracing {
// Sets the tracing output to the specified stream, closing the previous one
// if present.
// This will write a new trace header to the stream and write any
// subsequent commands.
// Passing nullptr will disable tracing.
void setTracing(llvm::raw_ostream *OS = nullptr);
} // namespace revng::tracing
