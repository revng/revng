#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/RawContainer.h"

namespace revng::pypeline {

using PTMLCBytesContainer = BytesContainer<"PTMLCBytesContainer",
                                           "text/x.c+ptml">;

using PTMLCFunctionBytesContainer = FunctionToBytesContainer<"PTMLCFunction"
                                                             "BytesContainer",
                                                             "text/x.c+ptml">;

using PTMLCTypeBytesContainer = TypeDefinitionToBytesContainer<"PTMLCType"
                                                               "BytesContainer",
                                                               "text/x.c+ptml">;

using RecompilableArchiveContainer = BytesContainer<"RecompilableArchive"
                                                    "Container",
                                                    "application/x-object">;

} // namespace revng::pypeline
