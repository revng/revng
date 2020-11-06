//
// Copyright (c) rev.ng Srls 2017-2020.
//

// LLVM includes
#include <llvm/Support/Path.h>

#include "revng-c/DecompilerResourceFinder/ResourceFinder.h"

namespace revng {

namespace c {

class PathList ResourceFinder({

#ifdef BUILD_PATH
  BUILD_PATH,
#endif

  llvm::sys::path::parent_path(getCurrentExecutableFullPath()).str(),

#ifdef INSTALL_PATH
  INSTALL_PATH,
#endif

});

} // end namespace c

} // end namespace revng
