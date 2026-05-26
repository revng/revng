#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/Support/CommandLine.h"

enum class DebugInfoLevel {
  No,
  Yes,
  IgnoreLibraries
};

struct ImporterOptions {
  const uint64_t BaseAddress;

  const DebugInfoLevel DebugInfo;
  const bool EnableRemoteDebugInfo;
};

[[nodiscard]] const ImporterOptions importerOptions();

extern llvm::cl::opt<uint64_t> BaseAddress;
extern llvm::cl::opt<DebugInfoLevel> DebugInfo;
extern llvm::cl::opt<bool> EnableRemoteDebugInfo;
