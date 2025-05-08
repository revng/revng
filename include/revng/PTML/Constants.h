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

namespace c {

namespace tokens {

inline constexpr llvm::StringRef Constant = "c.constant";
inline constexpr llvm::StringRef Directive = "c.directive";
inline constexpr llvm::StringRef Field = "c.field";
inline constexpr llvm::StringRef Function = "c.function";
inline constexpr llvm::StringRef FunctionParameter = "c.function_parameter";
inline constexpr llvm::StringRef Keyword = "c.keyword";
inline constexpr llvm::StringRef Operator = "c.operator";
inline constexpr llvm::StringRef StringLiteral = "c.string_literal";
inline constexpr llvm::StringRef Type = "c.type";
inline constexpr llvm::StringRef Variable = "c.variable";
inline constexpr llvm::StringRef GotoLabel = "c.goto_label";

} // namespace tokens

namespace scopes {

inline constexpr llvm::StringRef Function = "c.function";
inline constexpr llvm::StringRef FunctionBody = "c.function_body";
inline constexpr llvm::StringRef Scope = "c.scope";
inline constexpr llvm::StringRef StructBody = "c.struct";
inline constexpr llvm::StringRef UnionBody = "c.union";
inline constexpr llvm::StringRef TypeDeclarationsList = "c.type_declarations_"
                                                        "list";
inline constexpr llvm::StringRef FunctionDeclarationsList = "c.function_"
                                                            "declarations_list";
inline constexpr llvm::StringRef DynamicFunctionDeclarationsList = "c.dynamic_"
                                                                   "function_"
                                                                   "declaration"
                                                                   "s_list";
inline constexpr llvm::StringRef SegmentDeclarationsList = "c.segment_"
                                                           "declarations_list";

} // namespace scopes

} // namespace c

} // namespace ptml
