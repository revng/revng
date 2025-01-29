//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "CliftParser.h"

using namespace mlir::clift;

static thread_local llvm::SmallPtrSet<void *, 8> AsmRecursionMap;

llvm::SmallPtrSet<void *, 8> &mlir::clift::getAsmRecursionMap() {
  return AsmRecursionMap;
}
