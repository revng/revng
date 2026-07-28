//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include "clang/Driver/Driver.h"

#include "revng/ClangToModel/CompileFlags.h"
#include "revng/Support/Assert.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ResourceFinder.h"

using namespace llvm;

static std::optional<std::string> findHeaderDirectory(StringRef File) {
  auto MaybeHeaderPath = revng::ResourceFinder.findFile(File);
  if (not MaybeHeaderPath)
    return std::nullopt;
  auto Index = MaybeHeaderPath->rfind('/');
  if (Index == std::string::npos)
    return std::nullopt;

  return MaybeHeaderPath->substr(0, Index);
}

std::vector<std::string> revng::getClangCompileFlags() {
  StringRef CompileFlagsPath = "share/revng/compile-flags.cfg";
  auto MaybeCompileCFGPath = revng::ResourceFinder.findFile(CompileFlagsPath);
  revng_assert(MaybeCompileCFGPath);

  std::vector<std::string> Result;
  Result.push_back("--config=" + *MaybeCompileCFGPath);
  Result.push_back("-xc");

  SmallString<16> CompilerHeadersPath;
  {
    StringRef LLVMLibrary = getLibrariesFullPath().at("libLLVMSupport");
    using namespace llvm::sys::path;
    SmallString<16> ClangPath;
    append(ClangPath, parent_path(parent_path(LLVMLibrary)));
    append(ClangPath, Twine("bin"));
    append(ClangPath, Twine("clang"));
    CompilerHeadersPath = clang::driver::Driver::GetResourcesPath(ClangPath);
    append(CompilerHeadersPath, Twine("include"));
  }
  Result.push_back("-I" + CompilerHeadersPath.str().str());

  const char *PrimitivesHeader = "share/revng/include/primitive-types.h";
  auto MaybePrimitivesDir = findHeaderDirectory(PrimitivesHeader);
  revng_assert(MaybePrimitivesDir);
  Result.push_back("-I" + *MaybePrimitivesDir);

  return Result;
}
