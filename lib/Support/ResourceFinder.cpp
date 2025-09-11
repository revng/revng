//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

#include "revng/Support/Assert.h"
#include "revng/Support/ResourceFinder.h"

namespace revng {

PathList ResourceFinder({
  getCurrentRoot().str(),

#ifdef INSTALL_PATH
  INSTALL_PATH,
#endif

});

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
