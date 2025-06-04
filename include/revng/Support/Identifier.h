#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Helpers for validating C-like language identifiers: [_a-zA-Z][_a-zA-Z0-9]*
// Only alphanumeric characters in the ASCII space are considered valid.

#include "llvm/ADT/StringRef.h"

inline bool isInitialIdentifierCharacter(char Character) {
  if (Character == '_')
    return true;
  if ('a' <= Character and Character <= 'z')
    return true;
  if ('A' <= Character and Character <= 'Z')
    return true;
  return false;
}

inline bool isIdentifierCharacter(char Character) {
  if (isInitialIdentifierCharacter(Character))
    return true;
  if ('0' <= Character and Character <= '9')
    return true;
  return false;
}

inline bool validateIdentifier(llvm::StringRef Identifier) {
  if (Identifier.empty())
    return false;
  if (not isInitialIdentifierCharacter(Identifier.front()))
    return false;
  for (char Character : Identifier.drop_front()) {
    if (not isIdentifierCharacter(Character))
      return false;
  }
  return true;
}

inline std::string sanitizeIdentifier(llvm::StringRef Identifier) {
  std::string SanitaryIdentifier = Identifier.str();

  for (char &Character : SanitaryIdentifier) {
    if (not isIdentifierCharacter(Character))
      Character = '_';
  }

  if (not SanitaryIdentifier.empty()
      and not isInitialIdentifierCharacter(SanitaryIdentifier.front()))
    SanitaryIdentifier.insert(SanitaryIdentifier.begin(), '_');

  return SanitaryIdentifier;
}
