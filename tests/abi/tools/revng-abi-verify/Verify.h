#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

enum ExitCode {
  Success = 0,
  FailedOpeningTheInputFile = 1,
  FailedOpeningTheArtifactFile = 2,
  FailedDeserializingTheModel = 3,
  FailedVerifyingTheModel = 4,

  OnlyContinuousStackArgumentsAreSupported = 5,
  UnknownRegister = 6,
  UnknownStackOffset = 7,
  ArgumentCouldNotBeLocated = 8,
  FoundUnexpectedReturnValue = 9,
  ExpectedReturnValueNotFound = 10,
  UnknownReturnValueRegister = 11,
  ReturnValueCouldNotBeLocated = 12,
  CombinedStackArgumentsSizeIsWrong = 13
};

const std::error_category &thisToolError();
#define ERROR(CODE, ...) \
  llvm::createStringError(std::error_code(CODE, thisToolError()), __VA_ARGS__)

llvm::Error verifyABI(const TupleTree<model::Binary> &Binary,
                      llvm::StringRef RuntimeArtifact,
                      model::ABI::Values ABI);
