#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipebox/Containers.h"
#include "revng/PipeboxCommon/CliftContainers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::analyses {

/// Rename, retype and comment the local variables and goto labels of a
/// decompiled function, naming each one as it appears in the decompiled code.
///
/// The analysis receives, as configuration, the address of a function and a
/// list of edits:
///
/// ```yaml
/// Function: "0x401af7:Code_x86_64"
/// Edits:
///   - Target: var_0
///     Rename: counter
///     Retype: "uint32_t"
///     Comment: how many entries are left
///   - Target: label_1
///     Rename: retry
/// ```
///
/// `Target` is the name the C backend emitted, which is unique within the
/// function. `Retype` is written as a C type-name and resolved against the
/// decompiler headers, so any shape is accepted: `uint32_t`, `struct foo *`,
/// `char [8]`. `Comment` is emitted above the declaration of the variable, or
/// above the label, and applies to both; `Retype` applies to a variable only.
///
/// This covers the same edits `RENAME:`/`RETYPE:` comments express in
/// \ref EditCBody, without submitting the body: they need no position in the
/// code, only the entity they name. `Comment` is the comment belonging to the
/// entity, which needs no position either; a comment on a statement does need
/// one, so those remain the business of \ref EditCBody.
///
/// A directive an edit leaves out keeps what the entity already has, so
/// renaming a variable does not drop the comment it carries.
///
/// Every edit is applied on its own. One that cannot be applied is dropped and
/// reported on the `edit-by-name` logger, leaving the others alone. A local
/// variable and a goto label are identified, in the model, by the addresses of
/// the instructions using them and by nothing else, so two of them used only by
/// the same instructions cannot be told apart; an edit aimed at one of those is
/// among the ones dropped.
class EditByName {
public:
  static constexpr llvm::StringRef Name = "edit-by-name";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const CliftFunctionContainer &Clift,
                  const PTMLCContainer &TypeAndGlobalHeader,
                  const PTMLCContainer &HelperHeader);
};

} // namespace revng::pypeline::analyses
