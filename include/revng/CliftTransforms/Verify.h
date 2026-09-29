#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftDialect.h"

namespace clift {

[[nodiscard]] inline bool
isStrictPassVerificationEnabled(mlir::MLIRContext *Context) {
  auto Dialect = Context->getLoadedDialect<CliftDialect>();
  revng_assert(Dialect != nullptr);
  return Dialect->isStrictPassVerificationEnabled();
}

[[nodiscard]] inline mlir::LogicalResult
verifyNonLegalized(FunctionOp Function) {
  if (Function->hasAttr("clift.legalized")) {
    if (isStrictPassVerificationEnabled(Function.getContext()))
      return Function->emitOpError() << "cannot be transformed once legalized.";
  }
  return mlir::success();
}

} // namespace clift
