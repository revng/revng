#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"

#include "revng/ADT/ConstexprString.h"
#include "revng/Model/Helpers.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/Doxygen.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"

#include "revng-c/Pipes/Ranks.h"
#include "revng-c/Support/PTML.h"
#include "revng-c/Support/TokenDefinitions.h"

namespace ptml {

class CBuilder : public ptml::MarkupBuilder {
  using Tag = ptml::Tag;

public:
  enum class Operator {
    PointerDereference,
    AddressOf,
    Arrow,
    Dot,
    Add,
    Sub,
    Mul,
    Div,
    Modulo,
    RShift,
    LShift,
    And,
    Or,
    Xor,
    CmpEq,
    CmpNeq,
    CmpGt,
    CmpGte,
    CmpLt,
    CmpLte,
    BoolAnd,
    BoolOr,
    BoolNot,
    Assign,
    BinaryNot,
    UnaryMinus,
  };

  enum class Keyword {
    Const,
    Case,
    Switch,
    While,
    Do,
    Default,
    Break,
    Continue,
    If,
    Else,
    Return,
    Typedef,
    Struct,
    Union,
    Enum,
  };

  enum class Scopes {
    Scope,
    Function,
    FunctionBody,
    StructBody,
    UnionBody,
    TypeDeclarations,
    FunctionDeclarations,
    DynamicFunctionDeclarations,
    SegmentDeclarations,
  };

  enum class Directive {
    Include,
    Pragma,
    Define,
    If,
    IfDef,
    IfNotDef,
    ElIf,
    EndIf,
    Attribute,
  };

public:
  CBuilder(bool GeneratePlainC = false) : ptml::MarkupBuilder(GeneratePlainC) {}

private:
  llvm::StringRef toString(Keyword TheKeyword) const {
    switch (TheKeyword) {
    case Keyword::Const:
      return "const";
    case Keyword::Case:
      return "case";
    case Keyword::Switch:
      return "switch";
    case Keyword::While:
      return "while";
    case Keyword::Do:
      return "do";
    case Keyword::Default:
      return "default";
    case Keyword::Break:
      return "break";
    case Keyword::Continue:
      return "continue";
    case Keyword::If:
      return "if";
    case Keyword::Else:
      return "else";
    case Keyword::Return:
      return "return";
    case Keyword::Typedef:
      return "typedef";
    case Keyword::Struct:
      return "struct";
    case Keyword::Union:
      return "union";
    case Keyword::Enum:
      return "enum";
    default:
      revng_unreachable("Unknown keyword");
    }
  }

  llvm::StringRef toString(Operator OperatorOp) const {
    switch (OperatorOp) {
    case Operator::PointerDereference: {
      return "*";
    }
    case Operator::AddressOf: {
      if (!isGenerateTagLessPTML())
        return "&amp;";
      return "&";
    }

    case Operator::Arrow: {
      if (!isGenerateTagLessPTML())
        return "-&gt;";
      return "->";
    }

    case Operator::Dot: {
      return ".";
    }
    case Operator::Add: {
      return "+";
    }
    case Operator::Sub: {
      return "-";
    }
    case Operator::Mul: {
      return "*";
    }
    case Operator::Div: {
      return "/";
    }
    case Operator::Modulo: {
      return "%";
    }

    case Operator::RShift: {
      if (!isGenerateTagLessPTML())
        return "&gt;&gt;";
      return ">>";
    }

    case Operator::LShift: {
      if (!isGenerateTagLessPTML())
        return "&lt;&lt;";
      return "<<";
    }

    case Operator::And: {
      if (!isGenerateTagLessPTML())
        return "&amp;";
      return "&";
    }

    case Operator::Or: {
      return "|";
    }
    case Operator::Xor: {
      return "^";
    }
    case Operator::CmpEq: {
      return "==";
    }
    case Operator::CmpNeq: {
      return "!=";
    }
    case Operator::CmpGt: {
      if (!isGenerateTagLessPTML())
        return "&gt;";
      return ">";
    }
    case Operator::CmpGte: {
      if (!isGenerateTagLessPTML())
        return "&gt;=";
      return ">=";
    }
    case Operator::CmpLt: {
      if (!isGenerateTagLessPTML())
        return "&lt;";
      return "<";
    }

    case Operator::CmpLte: {
      if (!isGenerateTagLessPTML())
        return "&lt;=";
      return "<=";
    }
    case Operator::BoolAnd: {
      if (!isGenerateTagLessPTML())
        return "&amp;&amp;";
      return "&&";
    }
    case Operator::BoolOr: {
      return "||";
    }
    case Operator::BoolNot: {
      return "!";
    }
    case Operator::Assign: {
      return "=";
    }
    case Operator::BinaryNot: {
      return "~";
    }
    case Operator::UnaryMinus: {
      return "-";
    }
    default: {
      revng_unreachable("Unknown operator");
    }
    }
  }

  llvm::StringRef toString(Directive TheDirective) const {
    switch (TheDirective) {
    case Directive::Include:
      return "#include";
    case Directive::Pragma:
      return "#pragma";
    case Directive::Define:
      return "#define";
    case Directive::If:
      return "#if";
    case Directive::IfDef:
      return "#ifdef";
    case Directive::IfNotDef:
      return "#ifndef";
    case Directive::ElIf:
      return "#elif";
    case Directive::EndIf:
      return "#endif";
    case Directive::Attribute:
      return "__attribute__";
    default:
      revng_unreachable("Unknown directive");
    }
  }

  Tag operatorTagHelper(const llvm::StringRef Str) const {
    return tokenTag(Str, ptml::c::tokens::Operator);
  }

  std::string hexHelper(uint64_t Int) const {
    std::string Result;
    llvm::raw_string_ostream Out(Result);
    Out.write_hex(Int);
    Out.flush();
    return Result;
  }

  Tag keywordTagHelper(const llvm::StringRef Str) const {
    return ptml::MarkupBuilder::getTag(ptml::tags::Span, Str)
      .addAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
  }

  Tag directiveTagHelper(const llvm::StringRef Str) const {
    return tokenTag(Str, ptml::c::tokens::Directive);
  }

public:
  // Operators.
  Tag getOperator(Operator OperatorOp) const {
    return operatorTagHelper(toString(OperatorOp));
  }

  // Constants.
  Tag getConstantTag(const llvm::StringRef Str) const {
    return tokenTag(Str, ptml::c::tokens::Constant);
  }

  Tag getZeroTag() const { return getConstantTag("0"); }

  Tag getNullTag() const { return getConstantTag("NULL"); }

  Tag getTrueTag() const { return getConstantTag("true"); }

  Tag getFalseTag() const { return getConstantTag("false"); }

  Tag getHex(uint64_t Int) const {
    return getConstantTag("0x" + hexHelper(Int) + "U");
  }

  Tag getNumber(const llvm::APInt &I,
                unsigned int Radix = 10,
                bool Signed = false) const {
    llvm::SmallString<12> Result;
    I.toString(Result, Radix, Signed);
    if (I.getBitWidth() == 64 and I.isNegative())
      Result += 'U';

    return getConstantTag(Result);
  }

  template<class T>
  Tag getNumber(const T &I) const {
    return getConstantTag(std::to_string(I));
  }

  // String literal.
  Tag getStringLiteral(const llvm::StringRef Str) const {
    if (isGenerateTagLessPTML())
      return tokenTag(Str, ptml::c::tokens::StringLiteral);

    std::string Escaped;
    {
      llvm::raw_string_ostream EscapeHTMLStream(Escaped);
      llvm::printHTMLEscaped(Str, EscapeHTMLStream);
    }
    return tokenTag(Escaped, ptml::c::tokens::StringLiteral);
  }

  // Keywords.
  Tag getKeyword(Keyword TheKeyword) const {
    return keywordTagHelper(toString(TheKeyword));
  }

  // Scopes.
  Tag getScope(Scopes TheScope) const {
    switch (TheScope) {
    case Scopes::Scope:
      return scopeTag(ptml::c::scopes::Scope);
    case Scopes::Function:
      return scopeTag(ptml::c::scopes::Function);
    case Scopes::FunctionBody:
      return scopeTag(ptml::c::scopes::FunctionBody);
    case Scopes::StructBody:
      return scopeTag(ptml::c::scopes::StructBody);
    case Scopes::UnionBody:
      return scopeTag(ptml::c::scopes::UnionBody);
    case Scopes::TypeDeclarations:
      return scopeTag(ptml::c::scopes::TypeDeclarationsList);
    case Scopes::FunctionDeclarations:
      return scopeTag(ptml::c::scopes::FunctionDeclarationsList);
    case Scopes::DynamicFunctionDeclarations:
      return scopeTag(ptml::c::scopes::DynamicFunctionDeclarationsList);
    case Scopes::SegmentDeclarations:
      return scopeTag(ptml::c::scopes::SegmentDeclarationsList);
    default:
      revng_unreachable("Unknown scope");
    }
  }

  // Directives.
  Tag getDirective(Directive TheDirective) const {
    return directiveTagHelper(toString(TheDirective));
  }

  // Helpers.
  std::string getPragmaOnce() const {
    return getDirective(Directive::Pragma) + " " + getConstantTag("once")
           + "\n";
  }

  std::string getIncludeAngle(const llvm::StringRef Str) {
    std::string TheStr = "<" + Str.str() + ">";
    return getDirective(Directive::Include) + " " + getStringLiteral(TheStr)
           + "\n";
  }

  std::string getIncludeQuote(const llvm::StringRef Str) {
    std::string TheStr = "\"" + Str.str() + "\"";
    return getDirective(Directive::Include) + " " + getStringLiteral(TheStr)
           + "\n";
  }

  std::string getBlockComment(const llvm::StringRef Str,
                              bool Newline = true) const {
    return tokenTag("/* " + Str.str() + " */", ptml::tokens::Comment)
           + (Newline ? "\n" : "");
  }

  std::string getLineComment(const llvm::StringRef Str) {
    revng_check(!Str.contains('\n'));
    return tokenTag("// " + Str.str(), ptml::tokens::Comment) + "\n";
  }

  Tag getNameTag(const model::TypeDefinition &T) const {
    return tokenTag(T.name().str().str(), ptml::c::tokens::Type);
  }

  // Locations.
  constexpr const char *getLocationAttribute(bool IsDefinition) const {
    return IsDefinition ? ptml::attributes::LocationDefinition :
                          ptml::attributes::LocationReferences;
  }

  std::string toString(const model::TypeDefinition &T) const {
    if (isGenerateTagLessPTML())
      return "";
    return pipeline::toString(revng::ranks::TypeDefinition, T.key());
  }

  std::string getLocation(bool IsDefinition,
                          const model::TypeDefinition &T,
                          llvm::ArrayRef<std::string> AllowedActions) const {
    auto Result = getNameTag(T);
    if (isGenerateTagLessPTML())
      return Result.toString();

    std::string Location = toString(T);
    Result.addAttribute(getLocationAttribute(IsDefinition), Location);
    Result.addAttribute(attributes::ActionContextLocation, Location);

    if (not AllowedActions.empty())
      Result.addListAttribute(attributes::AllowedActions, AllowedActions);

    return Result.toString();
  }

  std::string
  getLocationDefinition(const model::TypeDefinition &T,
                        llvm::ArrayRef<std::string> AllowedActions = {}) const {
    return getLocation(true, T, AllowedActions);
  }

  std::string
  getLocationReference(const model::TypeDefinition &T,
                       llvm::ArrayRef<std::string> AllowedActions = {}) const {
    return getLocation(false, T, AllowedActions);
  }

  std::string getLocationDefinition(const model::PrimitiveType &P) const {
    std::string CName = P.getCName();
    auto Result = tokenTag(CName, ptml::c::tokens::Type);
    if (isGenerateTagLessPTML())
      return Result.toString();

    std::string L = pipeline::toString(revng::ranks::PrimitiveType,
                                       P.getCName());
    Result.addAttribute(getLocationAttribute(true), L);
    Result.addAttribute(attributes::ActionContextLocation, L);

    return Result.toString();
  }

  std::string getLocationReference(const model::PrimitiveType &P) const {
    std::string CName = P.getCName();
    auto Result = tokenTag(CName, ptml::c::tokens::Type);
    if (isGenerateTagLessPTML())
      return Result.toString();

    std::string L = pipeline::toString(revng::ranks::PrimitiveType,
                                       P.getCName());
    Result.addAttribute(getLocationAttribute(false), L);
    Result.addAttribute(attributes::ActionContextLocation, L);

    return Result.toString();
  }

  std::string toString(const model::Segment &T) const {
    if (isGenerateTagLessPTML())
      return "";
    return pipeline::toString(revng::ranks::Segment, T.key());
  }

  Tag getNameTag(const model::Segment &S) const {
    return tokenTag(S.name(), ptml::c::tokens::Variable);
  }

  std::string getLocation(bool IsDefinition, const model::Segment &S) const {
    std::string Location = toString(S);
    return getNameTag(S)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(ptml::attributes::ActionContextLocation, Location)
      .toString();
  }

  std::string getLocationDefinition(const model::Segment &S) const {
    return getLocation(true, S);
  }

  std::string getLocationReference(const model::Segment &S) const {
    return getLocation(false, S);
  }

  std::string toString(const model::EnumDefinition &Enum,
                       const model::EnumEntry &Entry) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::toString(revng::ranks::EnumEntry, Enum.key(), Entry.key());
  }

  std::string toString(const model::StructDefinition &Struct,
                       const model::StructField &Field) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::toString(revng::ranks::StructField,
                              Struct.key(),
                              Field.key());
  }

  std::string toString(const model::UnionDefinition &Union,
                       const model::UnionField &Field) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::toString(revng::ranks::UnionField,
                              Union.key(),
                              Field.key());
  }

  Tag getNameTag(const model::EnumDefinition &Enum,
                 const model::EnumEntry &Entry) const {
    return tokenTag(Enum.entryName(Entry), ptml::c::tokens::Field);
  }

  std::string getLocation(bool IsDefinition,
                          const model::EnumDefinition &Enum,
                          const model::EnumEntry &Entry) const {
    std::string Location = toString(Enum, Entry);
    return getNameTag(Enum, Entry)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(ptml::attributes::ActionContextLocation, Location)
      .toString();
  }

  template<class Aggregate, class Field>
  Tag getNameTag(const Aggregate &, const Field &F) const {
    return tokenTag(F.name(), c::tokens::Field);
  }

  template<typename Aggregate, typename Field>
  std::string
  getLocation(bool IsDefinition, const Aggregate &A, const Field &F) const {
    std::string Location = toString(A, F);
    return getNameTag(A, F)
      .addAttribute(getLocationAttribute(IsDefinition), Location)
      .addAttribute(attributes::ActionContextLocation, Location)
      .toString();
  }

  template<typename Aggregate, typename Field>
  std::string getLocationDefinition(const Aggregate &A, const Field &F) const {
    return getLocation(true, A, F);
  }

  template<typename Aggregate, typename Field>
  std::string getLocationReference(const Aggregate &A, const Field &F) const {
    return getLocation(false, A, F);
  }

  template<model::EntityWithComment Type>
  std::string getModelComment(Type T) {
    return ptml::comment(*this, T, "///", 0, 80);
  }

  std::string getFunctionComment(const model::Function &Function,
                                 const model::Binary &Binary) {
    return ptml::functionComment(*this, Function, Binary, "///", 0, 80);
  }
};
} // namespace ptml

/// Simple RAII object for create a pair of string, this will, given
/// a raw_ostream, print the \p Open when the object is created and
/// the \p Close when the object goes out of scope.
template<ConstexprString Open, ConstexprString Close>
struct PairedScope {
private:
  llvm::raw_ostream &OS;

public:
  PairedScope(llvm::raw_ostream &OS) : OS(OS) { OS << *Open; }
  ~PairedScope() { OS << *Close; }
};

/// RAII object for handling c style braced scopes. This will,
/// in order, open a brace pair, apply the Scope (think scopes, function body,
/// struct definition etc.) and indent the IndentedOstream, allowing a
/// egyptian-style c for most braced constructs
struct Scope {
private:
  using Braces = PairedScope<"{", "}">;
  Braces BraceScope;
  ptml::ScopeTag ScopeTag;
  ptml::IndentedOstream::Scope IndentScope;

public:
  Scope(ptml::IndentedOstream &Out,
        const llvm::StringRef Attribute = ptml::c::scopes::Scope) :
    BraceScope(Out),
    ScopeTag(Out.getMarkupBuilder()
               .getTag(ptml::tags::Div)
               .addAttribute(ptml::attributes::Scope, Attribute)
               .scope(Out, true)),
    IndentScope(Out.scope()) {}
};

/// RAII object for creating a comment tag with opening and closing
/// strings (e.g. /* and */) behaves similarly to \see PairedScope but will
/// enclose the entire text in an PTML comment tag
template<ConstexprString Open, ConstexprString Close>
struct CommentScope {
private:
  ptml::MarkupBuilder Builder;
  ptml::ScopeTag ScopeTag;
  PairedScope<Open, Close> PairScope;

public:
  CommentScope(llvm::raw_ostream &OS, bool GeneratePlainC) :
    Builder(GeneratePlainC),
    ScopeTag(Builder.tokenTag("", ptml::tokens::Comment).scope(OS, false)),
    PairScope(OS) {}
};

namespace helpers {

// Prepackaged comment scopes for c
using BlockComment = CommentScope<"/* ", " */">;
using LineComment = CommentScope<"// ", ConstexprString{}>;

} // namespace helpers
