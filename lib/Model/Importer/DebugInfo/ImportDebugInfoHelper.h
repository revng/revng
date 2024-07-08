#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/ProgramRunner.h"

namespace {
inline int runFetchDebugInfoWithLevel(llvm::StringRef InputFileName) {
  return ::Runner.run("revng",
                      { "model", "fetch-debuginfo", InputFileName.str() });
}
} // namespace
