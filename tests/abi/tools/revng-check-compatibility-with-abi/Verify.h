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
  NoArgumentBytesProvided = 6,
  UnknownArgumentRegister = 7,
  InsufficientStackSize = 8,
  ArgumentCanNotBeLocated = 9,
  LeftoverArgumentRegistersDetected = 10,
  CombinedStackArgumentsSizeIsWrong = 11,

  FoundUnexpectedReturnValue = 12,
  UnknownReturnValueRegister = 13,
  ReturnValueCanNotBeLocated = 14,
  LeftoverReturnValueRegistersDetected = 15,
  ExpectedReturnValueNotFound = 16
};

const std::error_category &thisToolError();
#define ERROR(CODE, ...) \
  llvm::createStringError(std::error_code(CODE, thisToolError()), __VA_ARGS__)

llvm::Error verifyABI(const TupleTree<model::Binary> &Binary,
                      llvm::StringRef RuntimeArtifact,
                      model::ABI::Values ABI);
