#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/PTML/CTokenEmitter.h"
#include "revng/Support/CTarget.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"

namespace mlir::clift {

void decompile(FunctionOp Function,
               CTokenEmitter &Emitter,
               const TargetCImplementation &Target);

} // namespace mlir::clift
