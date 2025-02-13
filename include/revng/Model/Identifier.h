#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include "revng/ADT/KeyedObjectContainer.h"

namespace model {

class VerifyHelper;

/// \note Zero-sized identifiers are valid
class Identifier : public llvm::SmallString<16> {
public:
  using llvm::SmallString<16>::SmallString;
  using llvm::SmallString<16>::operator=;

public:
  static const Identifier Empty;

public:
  /// Produce a (locally) valid identifier from an arbitrary string
  static Identifier fromString(llvm::StringRef Name);

  /// Produce a string without any character that's invalid for an identifier
  ///
  /// \note: given that reserved keywords are not considered by this function,
  ///        it does not necessarily emit a valid identifier.
  static Identifier sanitize(llvm::StringRef Name);

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
