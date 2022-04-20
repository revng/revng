#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "clang/Frontend/FrontendAction.h"

namespace revng {
namespace c {

/// Default arguments for clang tools used by revng-c
extern const std::vector<std::string> ClangToolDefaultArgs;

} // namespace c
} // namespace revng

/// Mutex for concurrently running clang::tooling
extern std::mutex ClangToolingMutex;

/// Thread-safely run ToolAction on Code, with revng-c default arguments
bool runThreadSafeClangTool(std::unique_ptr<clang::FrontendAction> ToolAction,
                            const std::string &Code);

/// Thread-safely run ToolAction on Code, passing Args as cmdline options
bool runThreadSafeClangTool(std::unique_ptr<clang::FrontendAction> ToolAction,
                            const std::string &Code,
                            const std::vector<std::string> &Args);
