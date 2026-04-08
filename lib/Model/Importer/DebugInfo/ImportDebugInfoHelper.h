#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/ProgramRunner.h"

inline Logger DILogger("dwarf-importer");

inline int runFetchDebugInfo(llvm::StringRef FullPath, bool Verbose) {
  if (llvm::sys::Process::GetEnv("REVNG_NO_FETCH_DEBUG_INFO"))
    return 1;

  revng_assert(::Runner.isProgramAvailable("revng"));
  std::vector<std::string> Args{ "model", "fetch-debuginfo", FullPath.str() };
  if (Verbose)
    Args.insert(Args.begin(), "--verbose");
  return ::Runner.run("revng", Args);
}
