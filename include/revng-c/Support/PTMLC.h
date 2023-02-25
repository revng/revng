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
#include "revng/PTML/ModelHelpers.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/PTML.h"
#include "revng-c/Support/TokenDefinitions.h"

namespace operators {
using ptml::Tag;

inline Tag operatorTag(const llvm::StringRef Str) {
  return ptml::tokenTag(Str, ptml::c::tokens::Operator);
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
inline const Tag BinaryNot = operatorTag("~");
inline const Tag UnaryMinus = operatorTag("-");

} // namespace operators

namespace constants {
using ptml::Tag;

inline Tag constant(const llvm::StringRef Str) {
  return ptml::tokenTag(Str, ptml::c::tokens::Constant);
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
  llvm::SmallString<12> Result;
  I.toString(Result, Radix, Signed);
  return constant(Result);
}

inline Tag stringLiteral(const llvm::StringRef Str) {
  return ptml::tokenTag(Str, ptml::c::tokens::StringLiteral);
}

} // namespace constants

namespace keywords {
using ptml::Tag;

inline Tag keyword(const llvm::StringRef str) {
  return Tag(ptml::tags::Span, str)
    .addAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
}

inline const Tag Const = keyword("const");
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

using ptml::scopeTag;
using ptml::Tag;
namespace scopes = ptml::c::scopes;

namespace detail {

inline constexpr auto TDL = scopes::TypeDeclarationsList;
inline constexpr auto FDL = scopes::FunctionDeclarationsList;
inline constexpr auto DFDL = scopes::DynamicFunctionDeclarationsList;
inline constexpr auto SDL = scopes::SegmentDeclarationsList;

} // end namespace detail

inline const Tag Scope = scopeTag(scopes::Scope);
inline const Tag Function = scopeTag(scopes::Function);
inline const Tag FunctionBody = scopeTag(scopes::FunctionBody);
inline const Tag Struct = scopeTag(scopes::StructBody);
inline const Tag Union = scopeTag(scopes::UnionBody);
inline const Tag TypeDeclarations = scopeTag(detail::TDL);
inline const Tag FunctionDeclarations = scopeTag(detail::FDL);
inline const Tag DynamicFunctionDeclarations = scopeTag(detail::DFDL);
inline const Tag SegmentDeclarations = scopeTag(detail::SDL);

} // namespace scopeTags

namespace directives {
using ptml::Tag;

inline Tag directive(const llvm::StringRef Str) {
  return ptml::tokenTag(Str, ptml::c::tokens::Directive);
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
  return ptml::tokenTag("/* " + Str.str() + " */", ptml::tokens::Comment)
         + (Newline ? "\n" : "");
}

inline std::string lineComment(const llvm::StringRef Str) {
  revng_check(Str.find("\n") == llvm::StringRef::npos);
  return ptml::tokenTag("// " + Str.str(), ptml::tokens::Comment) + "\n";
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
    ScopeTag(ptml::tokenTag("", ptml::tokens::Comment).scope(OS, false)),
    PairScope(OS) {}
};

namespace helpers {

// Prepackaged comment scopes for c
using BlockComment = CommentScope<"/* ", " */">;
using LineComment = CommentScope<"// ", ConstexprString{}>;

} // namespace helpers

namespace ptml {

constexpr inline const char *locationAttribute(bool IsDefinition) {
  return IsDefinition ? attributes::LocationDefinition :
                        attributes::LocationReferences;
}

inline std::string serializeLocation(const model::Type &T) {
  return pipeline::serializedLocation(revng::ranks::Type, T.key());
}

inline Tag getNameTag(const model::Type &T) {
  constexpr const char *const FunctionTypedefPrefix = "function_type_";
  model::Identifier Name;

  // Prefix for function types
  if (llvm::isa<model::RawFunctionType>(T)
      or llvm::isa<model::CABIFunctionType>(T))
    Name.append(FunctionTypedefPrefix);

  // Primitive types have reserved names, using model::Identifier adds an
  // unwanted prefix
  if (llvm::isa<model::PrimitiveType>(T))
    Name.append(T.name());
  else
    Name.append(model::Identifier::fromString(T.name()));

  return ptml::tokenTag(Name.str().str(), ptml::c::tokens::Type);
}

template<bool IsDefinition>
inline std::string getLocation(const model::Type &T) {
  auto Result = getNameTag(T).addAttribute(locationAttribute(IsDefinition),
                                           serializeLocation(T));
  // non-primitive types are editable
  if (not llvm::isa<model::PrimitiveType>(&T))
    Result.addAttribute(attributes::ModelEditPath,
                        modelEditPath::getCustomNamePath(T));
  return Result.serialize();
}

inline std::string getLocationDefinition(const model::Type &T) {
  return getLocation<true>(T);
}

inline std::string getLocationReference(const model::Type &T) {
  return getLocation<false>(T);
}

inline std::string serializeLocation(const model::Segment &T) {
  return pipeline::serializedLocation(revng::ranks::Segment, T.key());
}

inline Tag getNameTag(const model::Segment &S) {
  return ptml::tokenTag(S.name(), ptml::c::tokens::Variable);
}

template<bool IsDefinition>
inline std::string getLocation(const model::Segment &S) {
  return getNameTag(S)
    .addAttribute(locationAttribute(IsDefinition), serializeLocation(S))
    .addAttribute(attributes::ModelEditPath,
                  modelEditPath::getCustomNamePath(S))
    .serialize();
}

inline std::string getLocationDefinition(const model::Segment &S) {
  return getLocation<true>(S);
}

inline std::string getLocationReference(const model::Segment &S) {
  return getLocation<false>(S);
}

inline std::string
serializeLocation(const model::EnumType &Enum, const model::EnumEntry &Entry) {
  return pipeline::serializedLocation(revng::ranks::EnumEntry,
                                      Enum.key(),
                                      Entry.key());
}

inline std::string serializeLocation(const model::StructType &Struct,
                                     const model::StructField &Field) {
  return pipeline::serializedLocation(revng::ranks::StructField,
                                      Struct.key(),
                                      Field.key());
}

inline std::string serializeLocation(const model::UnionType &Union,
                                     const model::UnionField &Field) {
  return pipeline::serializedLocation(revng::ranks::UnionField,
                                      Union.key(),
                                      Field.key());
}

inline Tag
getNameTag(const model::EnumType &Enum, const model::EnumEntry &Entry) {
  return ptml::tokenTag(Enum.name().str().str() + "_"
                          + Entry.CustomName().str().str(),
                        c::tokens::Field);
}

template<bool IsDefinition>
inline std::string
getLocation(const model::EnumType &Enum, const model::EnumEntry &Entry) {
  return getNameTag(Enum, Entry)
    .addAttribute(locationAttribute(IsDefinition),
                  serializeLocation(Enum, Entry))
    .addAttribute(attributes::ModelEditPath,
                  modelEditPath::getCustomNamePath(Enum, Entry))
    .serialize();
}

// clang-format off
template<typename Field>
concept ModelStructOrUnionField =
  std::same_as<Field, model::StructField>
  or std::same_as<Field, model::UnionField>;
// clang-format on

template<ModelStructOrUnionField Field>
inline Tag getNameTag(const Field &F) {
  return ptml::tokenTag(F.name(), c::tokens::Field);
}

template<typename Aggregate, typename Field>
concept ModelStructOrUnionWithField = (std::same_as<model::StructType,
                                                    Aggregate>
                                       and std::same_as<model::StructField,
                                                        Field>)
                                      or (std::same_as<model::UnionType,
                                                       Aggregate>
                                          and std::same_as<model::UnionField,
                                                           Field>);

template<bool IsDefinition, typename Aggregate, typename Field>
  requires ModelStructOrUnionWithField<Aggregate, Field>
inline std::string getLocation(const Aggregate &A, const Field &F) {
  return getNameTag(F)
    .addAttribute(locationAttribute(IsDefinition), serializeLocation(A, F))
    .addAttribute(attributes::ModelEditPath,
                  modelEditPath::getCustomNamePath(A, F))
    .serialize();
}

// clang-format off
template<typename Aggregate, typename Field>
concept ModelAggregateWithField =
  ModelStructOrUnionWithField<Aggregate, Field>
  or (std::same_as<model::EnumType, Aggregate>
      and std::same_as<model::EnumEntry, Field>);
// clang-format on

template<typename Aggregate, typename Field>
  requires ModelAggregateWithField<Aggregate, Field>
inline std::string getLocationDefinition(const Aggregate &A, const Field &F) {
  return getLocation<true>(A, F);
}

template<typename Aggregate, typename Field>
  requires ModelAggregateWithField<Aggregate, Field>
inline std::string getLocationReference(const Aggregate &A, const Field &F) {
  return getLocation<false>(A, F);
}

} // namespace ptml
