#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/TupleTreeContainer.h"
#include "revng/Yield/Function.h"

namespace revng::pypeline {

using AssemblyInternalContainer = TupleTreeContainer<yield::Function,
                                                     Kinds::Function,
                                                     "AssemblyInternalContaine"
                                                     "r">;

}
