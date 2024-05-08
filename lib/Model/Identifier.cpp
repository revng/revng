/// \file Identifier.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Model/Identifier.h"

using namespace model;

Identifier Identifier::fromString(llvm::StringRef Name) {
  revng_assert(not Name.empty());
  Identifier Result;

  // For reserved C keywords prepend a non-reserved prefix and we're done.
  if (ReservedKeywords.contains(Name)) {
    Result += PrefixForReservedNames;
    Result += Name;
    return Result;
  }

  const auto BecomesUnderscore = [](const char C) {
    return not std::isalnum(C) or C == '_';
  };

  // For invalid C identifiers prepend the our reserved prefix.
  if (std::isdigit(Name[0]) or BecomesUnderscore(Name[0])) {
    Result += PrefixForReservedNames;
  }

  // Append the rest of the name
  Result += Name;

  return sanitize(Result);
}

Identifier Identifier::sanitize(llvm::StringRef Name) {
  Identifier Result(Name);

  // Convert all non-alphanumeric chars to underscores
  for (char &C : Result)
    if (not std::isalnum(C))
      C = '_';

  return Result;
}
