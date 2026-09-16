#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <set>
#include <string>

#include "llvm/Support/Error.h"

#include "revng/ADT/SortedVector.h"
#include "revng/Clift/Clift.h"
#include "revng/Model/Binary.h"
#include "revng/Model/GotoLabel.h"
#include "revng/Model/LocalVariable.h"
#include "revng/Pipebox/Containers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/TemporaryFile.h"

#include "ClangParse.h"

namespace revng::editcbody {

/// The address sets that identify more than one local variable, or more than
/// one goto label, of a function.
///
/// Both are identified by the addresses of the instructions using them, so two
/// of them sharing a set cannot be told apart: a model entry located there is
/// picked up by whichever of them comes first (see
/// \ref model::Function::findByLocation), no matter which one it was meant
/// for.
struct AmbiguousLocations {
  std::set<SortedVector<MetaAddress>> Variables;
  std::set<SortedVector<MetaAddress>> Labels;
};

/// Gather the address sets that cannot be attributed to a single local variable
/// or a single goto label of \p Function.
AmbiguousLocations collectAmbiguousLocations(clift::FunctionOp Function);

/// Build the model entry renaming, retyping and/or commenting a local variable,
/// located by the addresses of the instructions that use it.
llvm::Expected<model::LocalVariable>
makeLocalVariableEdit(clift::LocalVariableOp Variable,
                      const std::optional<std::string> &NewName,
                      const std::optional<std::string> &NewTypeName,
                      const std::optional<std::string> &NewComment,
                      const ResolvedTypeMap &ResolvedTypes,
                      const AmbiguousLocations &Ambiguous);

/// Build the model entry renaming and/or commenting a goto label, located by
/// the addresses of the instructions that use the label.
///
/// \p Label must be a label: whether the edit sits on one at all is a question
/// about the statement carrying it, which the caller answers.
llvm::Expected<model::GotoLabel>
makeGotoLabelEdit(clift::MakeLabelOp Label,
                  const std::optional<std::string> &NewName,
                  const std::optional<std::string> &NewComment,
                  const AmbiguousLocations &Ambiguous);

/// Write, to a temporary file, the type/global header and the helper header, so
/// that C written against the decompiled code can be parsed by Clang. Both are
/// the tagless headers the pipeline already produced, so nothing is re-emitted
/// here.
llvm::Expected<TemporaryFile>
writeHeader(const revng::pypeline::PTMLCContainer &TypeAndGlobalHeader,
            const revng::pypeline::PTMLCContainer &HelperHeader);

} // namespace revng::editcbody
