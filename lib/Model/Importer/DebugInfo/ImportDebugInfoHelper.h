#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Support/ProgramRunner.h"

namespace {
int runFetchDebugInfoWithLevel(llvm::StringRef InputFileName) {
  return ::Runner.run("revng",
                      { "model", "fetch-debuginfo", InputFileName.str() });
}
} // namespace
