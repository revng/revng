#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

void verifyABI(const TupleTree<model::Binary> &Binary,
               llvm::StringRef RuntimeArtifact,
               model::ABI::Values ABI);
