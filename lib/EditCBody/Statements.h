#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Assert.h"

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

/// A description of a statement kind, article included, as used in the messages
/// reporting the annotations that could not be applied.
inline llvm::StringRef describe(StatementKind Kind) {
  switch (Kind) {
  case StatementKind::LocalVariableDeclaration:
    return "a local variable declaration";
  case StatementKind::Expression:
    return "an expression statement";
  case StatementKind::Return:
    return "a `return` statement";
  case StatementKind::If:
    return "an `if` statement";
  case StatementKind::While:
    return "a `while` statement";
  case StatementKind::DoWhile:
    return "a `do`/`while` statement";
  case StatementKind::For:
    return "a `for` statement";
  case StatementKind::Switch:
    return "a `switch` statement";
  case StatementKind::Goto:
    // One kind covers `goto`, `break_to` and `continue_to`: the latter two are
    // macros for `goto`, so the C parse cannot tell them apart either.
    return "a jump statement";
  case StatementKind::Break:
    return "a `break` statement";
  case StatementKind::Continue:
    return "a `continue` statement";
  case StatementKind::Label:
    return "a label";
  case StatementKind::Case:
    return "a `case` label";
  case StatementKind::Default:
    return "a `default` label";
  }
  revng_abort("Invalid StatementKind");
}

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
