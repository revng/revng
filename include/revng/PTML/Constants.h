#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

namespace ptml {

namespace tags {

inline constexpr llvm::StringRef Div = "div";
inline constexpr llvm::StringRef Span = "span";

} // namespace tags

namespace attributes {

inline constexpr llvm::StringRef Scope = "data-scope";
inline constexpr llvm::StringRef Token = "data-token";
inline constexpr llvm::StringRef LocationDefinition = "data-location-"
                                                      "definition";
inline constexpr llvm::StringRef LocationReferences = "data-location-"
                                                      "references";
inline constexpr llvm::StringRef ActionContextLocation = "data-action-context-"
                                                         "location";
inline constexpr llvm::StringRef AllowedActions = "data-allowed-actions";

} // namespace attributes

namespace actions {

inline constexpr llvm::StringRef Rename = "rename";
inline constexpr llvm::StringRef Comment = "comment";
inline constexpr llvm::StringRef CodeSwitch = "codeSwitch";
inline constexpr llvm::StringRef EditType = "editType";

} // namespace actions

namespace scopes {} // namespace scopes

namespace tokens {

inline constexpr llvm::StringRef Indentation = "indentation";
inline constexpr llvm::StringRef Comment = "comment";

} // namespace tokens

} // namespace ptml
