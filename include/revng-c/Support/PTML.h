#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

namespace ptml::c {

namespace tokenTypes {

inline constexpr auto Function = "c.function";
inline constexpr auto Type = "c.type";
inline constexpr auto Operator = "c.operator";
inline constexpr auto Comparison = "c.comparison";
inline constexpr auto FunctionParameter = "c.function_parameter";
inline constexpr auto Variable = "c.variable";
inline constexpr auto Field = "c.field";
inline constexpr auto Constant = "c.constant";
inline constexpr auto StringLiteral = "c.string_literal";
inline constexpr auto Keyword = "c.keyword";
inline constexpr auto Directive = "c.directive";

} // namespace tokenTypes

namespace scopes {

inline constexpr auto Function = "c.function";
inline constexpr auto FunctionBody = "c.function_body";
inline constexpr auto Scope = "c.scope";
inline constexpr auto StructBody = "c.struct";
inline constexpr auto UnionBody = "c.union";

} // namespace scopes

} // namespace ptml::c
