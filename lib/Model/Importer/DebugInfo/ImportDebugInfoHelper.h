#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"

namespace {

inline int runFetchDebugInfo(llvm::StringRef InputFileName) {
  revng_assert(::Runner.isProgramAvailable("revng"));
  return ::Runner.run("revng",
                      { "model", "fetch-debuginfo", InputFileName.str() });
}

} // namespace
