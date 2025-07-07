//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Registry.h"

NB_MODULE(_cpp_pypeline, m) {
  TheRegistry.callAll(m);
}
