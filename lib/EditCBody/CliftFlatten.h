#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "Statements.h"

namespace mlir {
class Region;
} // namespace mlir

namespace revng::editcbody {

/// Flatten a Clift region into an ordered list of statements matching the
/// emitted C, appending them to `Output`.
///
/// The list includes the statements the C backend synthesizes (a `break`
/// closing a fallthrough switch case, loop `break`/`continue` labels), so it
/// lines up one-to-one with the flattened C code.
void flattenCliftRegion(mlir::Region &Root,
                        std::vector<CliftStatement> &Output);

} // namespace revng::editcbody
