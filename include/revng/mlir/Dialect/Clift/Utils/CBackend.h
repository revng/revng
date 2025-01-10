#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Model/Binary.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace ptml {
class CTypeBuilder;
}

namespace mlir::clift {

struct PlatformInfo {
  uint8_t sizeof_char;
  uint8_t sizeof_short;
  uint8_t sizeof_int;
  uint8_t sizeof_long;
  uint8_t sizeof_longlong;
  uint8_t sizeof_float;
  uint8_t sizeof_double;
  uint8_t sizeof_pointer;
};

std::string decompile(FunctionOp Function,
                      const PlatformInfo &Platform,
                      ptml::CTypeBuilder &Builder);

} // namespace mlir::clift
