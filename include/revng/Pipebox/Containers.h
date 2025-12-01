#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

using PTMLCFunctionBytesContainer = FunctionToBytesContainer<"PTMLCFunctionByte"
                                                             "sContainer",
                                                             "text/x.c+ptml">;

}
