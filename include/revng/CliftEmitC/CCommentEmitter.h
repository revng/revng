#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Clift/Clift.h"
#include "revng/PTML/CDoxygenEmitter.h"

namespace mlir::clift {

class CCommentEmitter {
  ptml::CTokenEmitter &Tokens;

public:
  explicit CCommentEmitter(ptml::CTokenEmitter &Tokens) : Tokens(Tokens) {}

public:
  void emitComment(llvm::StringRef Content);
  void emitFunctionComment(mlir::clift::FunctionOp Function);
};

} // namespace mlir::clift
