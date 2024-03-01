#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PTML/Tag.h"

namespace tokenDefinition::types {

using StringToken = llvm::SmallString<128>;
using TypeString = StringToken;

} // namespace tokenDefinition::types

/// Utility structure, used mainly in the C backend when generating variable
/// names in conjunction with PTML
/// It will have 2 Fields:
/// \property Declaration : Contains the string that's to be used when declaring
/// the variable. For example:
/// \code{.c}
/// int foo;
///   //^^^-- This is a declaration of foo
/// \endcode
/// \property Use : Contains the string that's to be used when using the
/// variable, such as:
/// \code{.c}
/// foo = bar + baz * bal;
/// // In this example foo, bar, baz and bal are "used"
/// \endcode
struct VariableTokens {
public:
  tokenDefinition::types::StringToken Declaration;
  tokenDefinition::types::StringToken Use;

public:
  VariableTokens() : Declaration(), Use() {}
  VariableTokens(const ptml::Tag &Declaration, const ptml::Tag &Use) :
    Declaration(Declaration.serialize()), Use(Use.serialize()) {}
  VariableTokens(const llvm::StringRef Declaration, const llvm::StringRef Use) :
    Declaration(Declaration), Use(Use) {}
  VariableTokens(const llvm::StringRef Use) : Use(Use) {}

  bool hasDeclaration() const { return !Declaration.empty(); }
};

/// Addition to VariableTokens that also preserves the 'naked' variable name
/// in the \property VariableName
/// Used mainly with helper structs where the struct's name is otherwise not
/// trivially obtainable
struct VariableTokensWithName : public VariableTokens {
public:
  tokenDefinition::types::StringToken VariableName;

public:
  template<typename T>
  VariableTokensWithName(const llvm::StringRef VariableName,
                         const T Declaration,
                         const T Use) :
    VariableTokens(Declaration, Use), VariableName(VariableName) {}
};
