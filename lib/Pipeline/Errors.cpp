/// \file Errors.cpp
/// \brief this file contains the various errors that can be thrown from whitin
/// a pipeline

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Errors.h"

using namespace llvm;
using namespace Pipeline;

char UnsatisfiableRequestError::ID;

void UnsatisfiableRequestError::log(raw_ostream &OS) const {
  OS << "While trying to deduce requirements of request:\n";
  Requested.dump(OS, 1);
  OS << "Could not find a rule to produce:\n";
  LastResort.dump(OS, 1);
}

std::error_code UnsatisfiableRequestError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

char UnknownTargetError::ID;

void UnknownTargetError::log(raw_ostream &OS) const {
  OS << "Could not erase\n";
  for (const auto &Name : Unknown) {
    Name.dump(OS);
    OS << "\n";
  }
  OS << "from: " << ContainerName << "\n";
  OS << "Because the queried backing container did not had it";
}

std::error_code UnknownTargetError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}
