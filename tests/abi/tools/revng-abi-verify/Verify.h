#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

enum ExitCode {
  Success = 0,
  FailedOpeningTheInputFile = 1,
  FailedOpeningTheOutputFile = 2,
  FailedOpeningTheArtifactFile = 3,
  FailedDeserializingTheModel = 4,
  FailedVerifyingTheModel = 5,
  FailedArgumentCountCheck = 6,
  FailedArgumentCompatibilityCheck = 7,
  FailedLocatingAnArgument = 8,
  FailedSelectingSingleArgumentLocation = 9
};

ExitCode verifyABI(const TupleTree<model::Binary> &Binary,
                   llvm::StringRef RuntimeArtifact,
                   model::ABI::Values ABI,
                   llvm::raw_fd_ostream &OutputStream);
