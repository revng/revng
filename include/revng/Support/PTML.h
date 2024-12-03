#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/Tag.h"

namespace ptml::c {

namespace tokens {

inline constexpr auto Constant = "c.constant";
inline constexpr auto Directive = "c.directive";
inline constexpr auto Field = "c.field";
inline constexpr auto Function = "c.function";
inline constexpr auto FunctionParameter = "c.function_parameter";
inline constexpr auto Keyword = "c.keyword";
inline constexpr auto Operator = "c.operator";
inline constexpr auto StringLiteral = "c.string_literal";
inline constexpr auto Type = "c.type";
inline constexpr auto Variable = "c.variable";
inline constexpr auto GotoLabel = "c.goto_label";

} // namespace tokens

namespace scopes {

inline constexpr auto Function = "c.function";
inline constexpr auto FunctionBody = "c.function_body";
inline constexpr auto Scope = "c.scope";
inline constexpr auto StructBody = "c.struct";
inline constexpr auto UnionBody = "c.union";
inline constexpr auto TypeDeclarationsList = "c.type_declarations_list";
inline constexpr auto FunctionDeclarationsList = "c.function_declarations_"
                                                 "list";
inline constexpr auto DynamicFunctionDeclarationsList = "c.dynamic_"
                                                        "function_"
                                                        "declaration"
                                                        "s_list";
inline constexpr auto SegmentDeclarationsList = "c.segment_"
                                                "declarations_"
                                                "list";

} // namespace scopes

} // namespace ptml::c
