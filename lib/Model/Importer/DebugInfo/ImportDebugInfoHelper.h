#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ProgramRunner.h"

namespace {

inline int runFetchDebugInfo(llvm::StringRef InputFileName, bool Verbose) {
  revng_assert(::Runner.isProgramAvailable("revng"));
  std::vector<std::string> Args{ "model",
                                 "fetch-debuginfo",
                                 InputFileName.str() };
  if (Verbose)
    Args.insert(Args.begin(), "--verbose");
  return ::Runner.run("revng", Args);
}

} // namespace
