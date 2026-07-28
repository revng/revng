#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

/// Import the annotations the user wrote in the body of a decompiled function.
///
/// The analysis receives, as configuration, the address of a function and a
/// piece of C code that must be structurally identical to the `emit-c` artifact
/// of that function, except for the comments the user added or edited. The C
/// code is parsed with Clang and matched, statement by statement, against the
/// Clift of the function; each statement is identified by the addresses of the
/// machine instructions it was lifted from.
///
/// Two kinds of annotation are recognized, both written as comments on the
/// line(s) preceding a statement. A comment trailing a statement on the same
/// line, after the code, is ignored: only comments that begin their own line
/// are considered.
///
/// - a plain comment becomes a `StatementComment` attached to the following
///   statement, located by the addresses of the instructions that make it up.
///   It can be placed before any statement that carries at least one address
///   (a computed expression, a `return`, or the condition of an `if` or a
///   loop); a statement with no address, such as a synthesized `break`, cannot
///   be commented.
///
/// - a `RENAME: <name>` and/or `RETYPE: <type>` comment renames and/or retypes
///   a local variable, recorded as a `LocalVariable` located by the addresses
///   of the instructions that use it. It can only be placed before the
///   variable's declaration.
class EditCBody {
public:
  static constexpr llvm::StringRef Name = "edit-c-body";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const CliftFunctionContainer &Clift,
                  const PTMLCContainer &TypeAndGlobalHeader,
                  const PTMLCContainer &HelperHeader);
};

} // namespace revng::pypeline::analyses
