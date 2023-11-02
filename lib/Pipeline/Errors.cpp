/// \file Errors.cpp
/// This file contains the various errors that can be thrown from within a
/// pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Errors.h"

using namespace llvm;
using namespace pipeline;

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

char AnnotatedError::ID;

void AnnotatedError::log(raw_ostream &OS) const {
  OS << ExtraData << "\n";
  std::string Indented = Inner;
  replaceAll(Indented, "\n", "\n  ");
  OS << "  " << Indented;
}

std::error_code AnnotatedError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}
