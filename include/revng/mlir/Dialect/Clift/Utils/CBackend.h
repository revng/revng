#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Model/Binary.h"
#include "revng/Support/CTarget.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace ptml {
class ModelCBuilder;
}

namespace mlir::clift {

std::string decompile(FunctionOp Function,
                      const TargetCImplementation &Target,
                      ptml::ModelCBuilder &Builder);

} // namespace mlir::clift
