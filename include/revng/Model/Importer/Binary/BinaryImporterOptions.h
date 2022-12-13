#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Support/CommandLine.h"

inline llvm::cl::list<std::string> ImportDebugInfo("import-debug-info",
                                                   llvm::cl::desc("path"),
                                                   llvm::cl::ZeroOrMore,
                                                   llvm::cl::cat(MainCategory));

#define DESCRIPTION                                                      \
  "Fetch debug info from canonical places or web. The dependency level " \
  "should be greater than 1 in order to process the dependency libraries."
using OptLevel = llvm::cl::opt<unsigned>;
inline OptLevel FetchDebugInfoWithLevel("fetch-debuginfo",
                                        llvm::cl::desc(DESCRIPTION),
                                        llvm::cl::value_desc("dependency "
                                                             "level"),
                                        llvm::cl::cat(MainCategory),
                                        llvm::cl::init(0));
#undef DESCRIPTION

#define DESCRIPTION "base address where dynamic objects should be loaded"
inline llvm::cl::opt<uint64_t> BaseAddress("base",
                                           llvm::cl::desc(DESCRIPTION),
                                           llvm::cl::value_desc("address"),
                                           llvm::cl::cat(MainCategory),
                                           llvm::cl::init(0x400000));
#undef DESCRIPTION
