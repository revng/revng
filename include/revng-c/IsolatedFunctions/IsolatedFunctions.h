#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <string>

// Forward declarations

namespace llvm {
class Function;
}

namespace model {
class Binary;
}

/// \brief Checks if Model has an isolated function with name F.getName(),
/// returning true on success, false on failure.
bool hasIsolatedFunction(const model::Binary &Model, const llvm::Function *F);

/// \brief Checks if Model has an isolated function with name F.getName(),
/// returning true on success, false on failure.
inline bool
hasIsolatedFunction(const model::Binary &Model, const llvm::Function &F) {
  return hasIsolatedFunction(Model, &F);
}

/// \brief Checks if Model has an isolated function with name FName, returning
/// true on success, false on failure.
bool hasIsolatedFunction(const model::Binary &Model, const std::string &FName);
