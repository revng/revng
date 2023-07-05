#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include "revng/ADT/KeyedObjectContainer.h"
#include "revng/Model/VerifyHelper.h"

namespace model {

extern const std::set<llvm::StringRef> ReservedKeywords;
extern const std::set<llvm::StringRef> ReservedPrefixes;

/// \note Zero-sized identifiers are valid
class Identifier : public llvm::SmallString<16> {
public:
  using llvm::SmallString<16>::SmallString;
  using llvm::SmallString<16>::operator=;

public:
  static const Identifier Empty;

public:
  static Identifier fromString(llvm::StringRef Name) {
    revng_assert(not Name.empty());
    Identifier Result;

    // For reserved C keywords prepend a non-reserved prefix and we're done.
    if (ReservedKeywords.contains(Name)) {
      Result += "prefix_";
      Result += Name;
      return Result;
    }

    auto StartsWithReservedPrefix = [](llvm::StringRef Name) {
      for (const auto &Prefix : ReservedPrefixes) {
        if (Name.startswith(Prefix)) {
          return true;
        }
      }
      return false;
    };

    const auto BecomesUnderscore = [](const char C) {
      return not std::isalnum(C) or C == '_';
    };

    // For invalid C identifiers prepend the our reserved prefix.
    if (std::isdigit(Name[0]) or BecomesUnderscore(Name[0])
        or StartsWithReservedPrefix(Name))
      Result += "prefix_";

    // Append the rest of the name
    Result += Name;

    // Convert all non-alphanumeric chars to underscores
    for (char &C : Result)
      if (not std::isalnum(C))
        C = '_';

    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

} // namespace model

/// KeyedObjectTraits for std::string based on its value
template<>
struct KeyedObjectTraits<model::Identifier>
  : public IdentityKeyedObjectTraits<model::Identifier> {};

template<>
struct llvm::yaml::ScalarTraits<model::Identifier> {
  static void
  output(const model::Identifier &Value, void *, llvm::raw_ostream &Output) {
    Output << Value;
  }

  static StringRef
  input(llvm::StringRef Scalar, void *, model::Identifier &Value) {
    Value = model::Identifier(Scalar);
    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Double; }
};
