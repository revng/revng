//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"

#include "revng/Support/ResourceFinder.h"

using namespace llvm::sys::path;

namespace revng {

PathList ResourceFinder({
  parent_path(parent_path(getCurrentExecutableFullPath())).str(),

#ifdef INSTALL_PATH
  INSTALL_PATH,
#endif

});

} // namespace revng
