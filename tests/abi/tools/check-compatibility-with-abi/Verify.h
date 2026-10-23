#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <vector>

#include "llvm/ADT/StringRef.h"

#include "revng/Model/TypeDefinitionKind.h"

#include "ExpectedFailures.h"

std::vector<Failure>
verifyABI(const TupleTree<model::Binary> &Binary,
          llvm::StringRef RuntimeArtifact,
          model::ABI::Values ABI,
          std::optional<model::TypeDefinitionKind::Values> VerifyType);
