#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Binary.h"

#include "Statements.h"

namespace revng::editcbody {

/// The result of parsing the user's C: the flattened statements and, for each
/// `RETYPE:` directive, the model type it names (keyed by the directive text as
/// written). The types are resolved by Clang during the same parse, from
/// synthetic typedefs prepended to the code.
struct ParsedFunction {
  std::vector<CStatement> Statements;
  std::map<std::string, model::UpcastableType> ResolvedTypes;
};

/// Parse a single decompiled function definition, prefixed by the header at
/// `HeaderPath`, into a flattened list of statements with their leading
/// comments. `Binary` resolves the types named by `RETYPE:` directives.
///
/// This is the only entry point into the Clang-based parsing, so that Clang
/// stays confined to its own translation unit.
llvm::Expected<ParsedFunction> parseUserFunction(llvm::StringRef HeaderPath,
                                                 llvm::StringRef CCode,
                                                 const model::Binary &Binary);

} // namespace revng::editcbody
