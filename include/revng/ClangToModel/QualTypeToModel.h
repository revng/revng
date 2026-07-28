#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "clang/AST/Type.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Binary.h"

namespace clang {
class ASTContext;
} // namespace clang

namespace revng {

/// Convert a Clang `QualType` into the corresponding model type, resolving
/// named definitions (structs, unions, enums, function prototypes) against
/// `Binary` and mapping builtins to `model::PrimitiveType`. Any diagnostic is
/// appended to `Errors`, each line prefixed with `ErrorPrefix`. On failure an
/// empty `model::UpcastableType` is returned.
RecursiveCoroutine<model::UpcastableType>
qualTypeToModel(const clang::QualType &QT,
                const model::Binary &Binary,
                clang::ASTContext &Context,
                std::vector<std::string> &Errors,
                llvm::StringRef ErrorPrefix);

} // namespace revng
