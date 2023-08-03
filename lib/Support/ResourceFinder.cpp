//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Path.h"

#include "revng/Support/ResourceFinder.h"

namespace revng {

PathList ResourceFinder({
  getCurrentRoot().str(),

#ifdef INSTALL_PATH
  INSTALL_PATH,
#endif

});

} // namespace revng
