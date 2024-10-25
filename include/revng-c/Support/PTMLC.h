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
#include "revng/Pipes/Ranks.h"

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
  CBuilder(ptml::MarkupBuilder B = {}) : ptml::MarkupBuilder(B) {}
  CBuilder(bool EnableTaglessMode) :
    ptml::MarkupBuilder{ .IsInTaglessMode = EnableTaglessMode } {}

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
      if (not IsInTaglessMode)
        return "&amp;";
      return "&";
    }

    case Operator::Arrow: {
      if (not IsInTaglessMode)
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
      if (not IsInTaglessMode)
        return "&gt;&gt;";
      return ">>";
    }

    case Operator::LShift: {
      if (not IsInTaglessMode)
        return "&lt;&lt;";
      return "<<";
    }

    case Operator::And: {
      if (not IsInTaglessMode)
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
      if (not IsInTaglessMode)
        return "&gt;";
      return ">";
    }
    case Operator::CmpGte: {
      if (not IsInTaglessMode)
        return "&gt;=";
      return ">=";
    }
    case Operator::CmpLt: {
      if (not IsInTaglessMode)
        return "&lt;";
      return "<";
    }

    case Operator::CmpLte: {
      if (not IsInTaglessMode)
        return "&lt;=";
      return "<=";
    }
    case Operator::BoolAnd: {
      if (not IsInTaglessMode)
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
    if (IsInTaglessMode)
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

  ptml::Tag getTypeKeyword(const model::TypeDefinition &T) const {
    if (llvm::isa<model::EnumDefinition>(T))
      return getKeyword(ptml::CBuilder::Keyword::Enum);

    else if (llvm::isa<model::StructDefinition>(T))
      return getKeyword(ptml::CBuilder::Keyword::Struct);

    else if (llvm::isa<model::UnionDefinition>(T))
      return getKeyword(ptml::CBuilder::Keyword::Union);

    else if (llvm::isa<model::TypedefDefinition>(T)
             || llvm::isa<model::RawFunctionDefinition>(T)
             || llvm::isa<model::CABIFunctionDefinition>(T))
      return getKeyword(ptml::CBuilder::Keyword::Typedef);

    else
      revng_abort("Unsupported type definition.");
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

public:
  static constexpr std::string_view structPaddingPrefix() {
    return "_padding_at_";
  }

  static constexpr std::string_view artificialReturnValuePrefix() {
    return "_artificial_struct_returned_by_";
  }
  static constexpr std::string_view artificialArrayWrapperPrefix() {
    return "_artificial_wrapper_";
  }

  static constexpr std::string_view artificialReturnValueFieldPrefix() {
    return "field_";
  }
  static constexpr std::string_view artificialArrayWrapperFieldName() {
    return "the_array";
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
  const ptml::MarkupBuilder &B;
  ptml::ScopeTag ScopeTag;
  PairedScope<Open, Close> PairScope;

public:
  CommentScope(llvm::raw_ostream &OS, const ptml::MarkupBuilder &B) :
    B(B),
    ScopeTag(B.tokenTag("", ptml::tokens::Comment).scope(OS, false)),
    PairScope(OS) {}
};

namespace helpers {

// Prepackaged comment scopes for c
using BlockComment = CommentScope<"/* ", " */">;
using LineComment = CommentScope<"// ", ConstexprString{}>;

} // namespace helpers
