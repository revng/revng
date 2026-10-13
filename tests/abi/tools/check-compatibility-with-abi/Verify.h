#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/TypeDefinitionKind.h"

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI,
               model::TypeDefinitionKind::Values ExpectedPrototypeKind);
