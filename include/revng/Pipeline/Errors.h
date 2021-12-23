#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"

namespace Pipeline {

/// A error thrown when the autoenforcer is not able to
/// satisfy the deduced requirements to produce a target.
class UnsatisfiableRequestError
  : public llvm::ErrorInfo<UnsatisfiableRequestError> {
private:
  ContainerToTargetsMap Requested;
  ContainerToTargetsMap LastResort;

public:
  static char ID;

  UnsatisfiableRequestError(ContainerToTargetsMap Requested,
                            ContainerToTargetsMap LastResort) :
    Requested(std::move(Requested)), LastResort(std::move(LastResort)) {}

  std::error_code convertToErrorCode() const override;
  void log(llvm::raw_ostream &OS) const override;
};

/// Error thrown when the user tries to erase a target
/// that does not exists.
class UnknownTargetError : public llvm::ErrorInfo<UnknownTargetError> {
private:
  TargetsList Unknown;
  std::string ContainerName;

public:
  static char ID;

  UnknownTargetError(TargetsList Unknown, llvm::StringRef ContainerName) :
    Unknown(std::move(Unknown)), ContainerName(ContainerName.str()) {}

  std::error_code convertToErrorCode() const override;
  void log(llvm::raw_ostream &OS) const override;
};

} // namespace Pipeline
