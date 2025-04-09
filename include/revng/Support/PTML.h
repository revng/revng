#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"

namespace ptml::c {

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

} // namespace ptml::c
