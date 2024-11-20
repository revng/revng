//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "CliftParser.h"

using namespace mlir::clift;

static thread_local std::map<uint64_t, void *> AsmRecursionMap;

std::map<uint64_t, void *> &mlir::clift::getAsmRecursionMap() {
  return AsmRecursionMap;
}
