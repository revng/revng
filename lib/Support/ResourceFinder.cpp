//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"

#include "revng/Support/ResourceFinder.h"

namespace revng {

PathList ResourceFinder({

#ifdef BUILD_PATH
  BUILD_PATH,
#endif

  llvm::sys::path::parent_path(getCurrentExecutableFullPath()).str(),

#ifdef INSTALL_PATH
  INSTALL_PATH,
#endif

});

} // namespace revng
