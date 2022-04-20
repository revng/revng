//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <utility>

#include "clang/Tooling/Tooling.h"

#include "revng-c/ThreadSafeClangTooling/ThreadSafeClangTooling.h"

namespace revng {
namespace c {

const std::vector<std::string> ClangToolDefaultArgs{ // C language
                                                     "-xc",
                                                     // C11 dialect
                                                     "-std=c11"
};

} // namespace c
} // namespace revng

std::mutex ClangToolingMutex;

bool runThreadSafeClangTool(std::unique_ptr<clang::FrontendAction> ToolAction,
                            const std::string &Code,
                            const std::vector<std::string> &Args) {
  std::scoped_lock ClangToolingGuard{ ClangToolingMutex };
  return clang::tooling::runToolOnCodeWithArgs(std::move(ToolAction),
                                               Code,
                                               Args);
}

bool runThreadSafeClangTool(std::unique_ptr<clang::FrontendAction> ToolAction,
                            const std::string &Code) {
  return runThreadSafeClangTool(std::move(ToolAction),
                                Code,
                                revng::c::ClangToolDefaultArgs);
}
