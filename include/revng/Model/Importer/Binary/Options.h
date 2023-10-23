#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
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

  const llvm::ArrayRef<std::string> AdditionalDebugInfoPaths;
};

[[nodiscard]] const ImporterOptions importerOptions();

extern llvm::cl::opt<uint64_t> BaseAddress;
extern llvm::cl::list<std::string> ImportDebugInfo;
extern llvm::cl::opt<DebugInfoLevel> DebugInfo;
extern llvm::cl::opt<bool> EnableRemoteDebugInfo;
