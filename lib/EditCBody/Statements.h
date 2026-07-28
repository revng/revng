#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace revng::editcbody {

/// The kind of a statement, coarse enough to be shared between the Clang AST
/// and the Clift representation.
enum class StatementKind {
  LocalVariableDeclaration,
  Expression,
  Return,
  If,
  While,
  DoWhile,
  For,
  Switch,
  Goto,
  Break,
  Continue,
  Label,
  Case,
  Default,
};

/// A statement of the user's C code, in the flattened pre-order walk.
struct CStatement {
  StatementKind Kind = {};
  unsigned BeginOffset = 0;
  llvm::SmallVector<std::string> LeadingComments;
};

/// A statement of the Clift function, in the flattened pre-order walk, matching
/// the one produced for the C code. `Op` is null for statements the C backend
/// synthesizes (the `break` closing a fallthrough switch case, and the labels
/// of loops with `break`/`continue`); they carry no address and cannot be
/// commented.
struct CliftStatement {
  StatementKind Kind = {};
  mlir::Operation *Op = nullptr;
};

} // namespace revng::editcbody
