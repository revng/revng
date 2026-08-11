//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ResourceFinder.h"

namespace revng {

// Build the search-path list. REVNG_RESOURCES (colon-separated, the
// usual PATH-style convention) lets callers point ResourceFinder at
// an extra root — e.g. a merged install tree assembled at test time
// that lives outside revng's own (nix-store) install path. Probed
// before the built-in roots so the override wins.
static std::vector<std::string> buildResourceFinderPaths() {
  std::vector<std::string> Result;
  if (auto Env = llvm::sys::Process::GetEnv("REVNG_RESOURCES")) {
    llvm::StringRef Remaining = *Env;
    while (not Remaining.empty()) {
      auto [Head, Tail] = Remaining.split(':');
      if (not Head.empty())
        Result.emplace_back(Head);
      Remaining = Tail;
    }
  }
  Result.emplace_back(getCurrentRoot().str());
#ifdef INSTALL_PATH
  Result.emplace_back(INSTALL_PATH);
#endif
#ifdef LIBTCG_PATH
  Result.emplace_back(LIBTCG_PATH);
#endif
  return Result;
}

PathList ResourceFinder(buildResourceFinderPaths());

std::string getComponentsHash() {
  std::string Directory = "share/revng/component-hashes";
  std::vector<std::string> Files = ResourceFinder.list(Directory, "");
  llvm::sort(Files);

  std::string Result;
  for (std::string &File : Files) {
    auto Buf = cantFail(errorOrToExpected(llvm::MemoryBuffer::getFile(File)));
    llvm::StringRef Contents = Buf->getBuffer().trim();
    Result.append(Contents.begin(), Contents.end());
  }

  return Result;
}

} // namespace revng
