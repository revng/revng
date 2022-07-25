#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"

#include "revng/ADT/ConstexprString.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"

#include "revng-c/Support/PTML.h"
#include "revng-c/Support/TokenDefinitions.h"

inline ptml::Tag
tokenTag(const llvm::StringRef Str, const llvm::StringRef Token) {
  return ptml::Tag(ptml::tags::Span, Str)
    .addAttribute(ptml::attributes::Token, Token);
}

namespace operators {
using ptml::Tag;

inline Tag operatorTag(const llvm::StringRef Str) {
  return tokenTag(Str, ptml::c::tokenTypes::Operator);
}

inline const Tag PointerDereference = operatorTag("*");
inline const Tag AddressOf = operatorTag("&amp;");
inline const Tag Arrow = operatorTag("-&gt;");
inline const Tag Dot = operatorTag(".");
inline const Tag Add = operatorTag("+");
inline const Tag Sub = operatorTag("-");
inline const Tag Mul = operatorTag("*");
inline const Tag Div = operatorTag("/");
inline const Tag Modulo = operatorTag("%");
inline const Tag RShift = operatorTag("&gt;&gt;");
inline const Tag LShift = operatorTag("&lt;&lt;");
inline const Tag And = operatorTag("&amp;");
inline const Tag Or = operatorTag("|");
inline const Tag Xor = operatorTag("^");
inline const Tag CmpEq = operatorTag("==");
inline const Tag CmpNeq = operatorTag("!=");
inline const Tag CmpGt = operatorTag("&gt;");
inline const Tag CmpGte = operatorTag("&gt;=");
inline const Tag CmpLt = operatorTag("&lt;");
inline const Tag CmpLte = operatorTag("&lt;=");
inline const Tag BoolAnd = operatorTag("&amp;&amp;");
inline const Tag BoolOr = operatorTag("||");
inline const Tag BoolNot = operatorTag("!");
inline const Tag Assign = operatorTag("=");

} // namespace operators

namespace constants {
using ptml::Tag;

inline Tag constant(const llvm::StringRef Str) {
  return tokenTag(Str, ptml::c::tokenTypes::Constant);
}

inline const Tag True = constant("true");
inline const Tag False = constant("false");
inline const Tag Null = constant("NULL");
inline const Tag Zero = constant("0");

template<typename T>
concept toStringAble = requires(const T &Var) {
  { std::to_string(Var) } -> std::same_as<std::string>;
};

template<toStringAble T>
inline Tag number(const T &I) {
  return constant(std::to_string(I));
}

inline Tag
number(const llvm::APInt &I, unsigned int Radix = 10, bool Signed = false) {
  return constant(I.toString(Radix, Signed));
}

inline Tag stringLiteral(const llvm::StringRef Str) {
  return tokenTag(Str, ptml::c::tokenTypes::StringLiteral);
}

} // namespace constants

namespace keywords {
using ptml::Tag;

inline Tag keyword(const llvm::StringRef str) {
  return Tag(ptml::tags::Span, str)
    .addAttribute(ptml::attributes::Token, ptml::c::tokenTypes::Keyword);
}

inline const Tag Case = keyword("case");
inline const Tag Switch = keyword("switch");
inline const Tag While = keyword("while");
inline const Tag Do = keyword("do");
inline const Tag Default = keyword("default");
inline const Tag Break = keyword("break");
inline const Tag Continue = keyword("continue");
inline const Tag If = keyword("if");
inline const Tag Else = keyword("else");
inline const Tag Return = keyword("return");
inline const Tag Typedef = keyword("typedef");
inline const Tag Struct = keyword("struct");
inline const Tag Union = keyword("union");
inline const Tag Enum = keyword("enum");

} // namespace keywords

namespace scopeTags {
using ptml::Tag;
namespace scopes = ptml::c::scopes;

inline Tag scopeTag(const llvm::StringRef Scope) {
  return Tag(ptml::tags::Div).addAttribute(ptml::attributes::Scope, Scope);
}

inline const Tag Scope = scopeTag(scopes::Scope);
inline const Tag Function = scopeTag(scopes::Function);
inline const Tag FunctionBody = scopeTag(scopes::FunctionBody);
inline const Tag Struct = scopeTag(scopes::StructBody);
inline const Tag Union = scopeTag(scopes::UnionBody);

} // namespace scopeTags

namespace directives {
using ptml::Tag;

inline Tag directive(const llvm::StringRef Str) {
  return tokenTag(Str, ptml::c::tokenTypes::Directive);
}

inline const Tag Include = directive("#include");
inline const Tag Pragma = directive("#pragma");
inline const Tag Define = directive("#define");
inline const Tag If = directive("#if");
inline const Tag IfDef = directive("#ifdef");
inline const Tag IfNotDef = directive("#ifndef");
inline const Tag ElIf = directive("#elif");
inline const Tag EndIf = directive("#endif");
inline const Tag Attribute = directive("__attribute__");

} // namespace directives

namespace helpers {

inline std::string pragmaOnce() {
  return directives::Pragma + " " + constants::constant("once") + "\n";
}

inline std::string includeAngle(const llvm::StringRef Str) {
  return directives::Include + " "
         + constants::stringLiteral("&lt;" + Str.str() + "&gt;") + "\n";
}

inline std::string includeQuote(const llvm::StringRef Str) {
  return directives::Include + " "
         + constants::stringLiteral("&quot;" + Str.str() + "&quot;") + "\n";
}

inline std::string
blockComment(const llvm::StringRef Str, bool Newline = true) {
  return tokenTag("/* " + Str.str() + " */", ptml::tokens::Comment)
         + (Newline ? "\n" : "");
}

inline std::string lineComment(const llvm::StringRef Str) {
  revng_check(Str.find("\n") == llvm::StringRef::npos);
  return tokenTag("// " + Str.str(), ptml::tokens::Comment) + "\n";
}

inline std::string attribute(const llvm::StringRef Str) {
  return directives::Attribute + "((" + Str.str() + "))";
}

inline std::string hex(uint64_t Int) {
  std::string Result;
  llvm::raw_string_ostream Out(Result);
  Out.write_hex(Int);
  Out.flush();
  return Result;
}

inline const auto Packed = helpers::attribute("packed");

} // namespace helpers

namespace constants {

inline Tag hex(uint64_t Int) {
  return constants::constant("0x" + helpers::hex(Int) + "U");
}

} // namespace constants

/// Simple RAII object for create a pair of string, this will
/// , given a raw_ostream, print the \p Open when the object is
/// created and the \p Close when the object goes out of scope
template<ConstexprString Open, ConstexprString Close>
struct PairedScope {
private:
  llvm::raw_ostream &OS;

public:
  PairedScope(llvm::raw_ostream &OS) : OS(OS) { OS << Open.String.data(); }
  ~PairedScope() { OS << Close.String.data(); }
};

/// RAII object for handling c style braced scopes. This will,
/// in order, open a brace pair, apply the Scope (think scopes, function body,
/// struct definition etc.) and indent the PTMLIndentedOstream, allowing a
/// egyptian-style c for most braced constructs
struct Scope {
private:
  using Braces = PairedScope<"{", "}">;
  Braces BraceScope;
  ptml::ScopeTag ScopeTag;
  ptml::PTMLIndentedOstream::Scope IndentScope;

public:
  Scope(ptml::PTMLIndentedOstream &Out,
        const ptml::Tag &TheTag = scopeTags::Scope) :
    BraceScope(Out),
    ScopeTag(TheTag.scope(Out, true)),
    IndentScope(Out.scope()) {}
};

/// RAII object for creating a comment tag with opening and closing
/// strings (e.g. /* and */) behaves similarly to \see PairedScope but will
/// enclose the entire text in an PTML comment tag
template<ConstexprString Open, ConstexprString Close>
struct CommentScope {
private:
  ptml::ScopeTag ScopeTag;
  PairedScope<Open, Close> PairScope;

public:
  CommentScope(llvm::raw_ostream &OS) :
    ScopeTag(tokenTag("", ptml::tokens::Comment).scope(OS, false)),
    PairScope(OS) {}
};

namespace helpers {

// Prepackaged comment scopes for c
using BlockComment = CommentScope<"/* ", " */">;
using LineComment = CommentScope<"// ", ConstexprString{}>;

} // namespace helpers
