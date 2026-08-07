#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"

#include "Statements.h"

namespace revng::editcbody {

/// The model type named by each `RETYPE:` directive, keyed by its text as
/// written.
using ResolvedTypeMap = std::map<std::string, model::UpcastableType>;

/// The result of parsing the user's C: the flattened statements and the types
/// named by the `RETYPE:` directives found in it, resolved by Clang during the
/// same parse from synthetic declarations prepended to the code.
struct ParsedFunction {
  std::vector<CStatement> Statements;
  ResolvedTypeMap ResolvedTypes;
};

/// Resolve C type names against the header at `HeaderPath`, into the model
/// types they denote. A name that does not denote a type resolves to an empty
/// type; the caller decides what to do about it.
///
/// The names are written as C type-names, so any shape is accepted: `uint32_t`,
/// `struct foo *`, `char [8]`.
llvm::Expected<ResolvedTypeMap>
resolveTypeNames(llvm::StringRef HeaderPath,
                 llvm::ArrayRef<std::string> TypeNames,
                 const model::Binary &Binary);

/// Parse a single decompiled function definition, prefixed by the header at
/// `HeaderPath`, into a flattened list of statements with their leading
/// comments. `Binary` resolves the types named by `RETYPE:` directives.
///
/// The body and the types are resolved by the same parse, so a function that
/// carries `RETYPE:` directives costs one Clang invocation, not two.
llvm::Expected<ParsedFunction> parseUserFunction(llvm::StringRef HeaderPath,
                                                 llvm::StringRef CCode,
                                                 const model::Binary &Binary);

} // namespace revng::editcbody
