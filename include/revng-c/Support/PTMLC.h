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

namespace ptml {

class PTMLCBuilder : public ptml::PTMLBuilder {
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
  PTMLCBuilder(bool GeneratePlainC = false) :
    ptml::PTMLBuilder(GeneratePlainC) {}

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
    return ptml::PTMLBuilder::getTag(ptml::tags::Span, Str)
      .addAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
  }

  Tag scopeTagHelper(const llvm::StringRef AttributeName) const {
    return ptml::PTMLBuilder::getTag(ptml::tags::Div)
      .addAttribute(ptml::attributes::Scope, AttributeName);
  }

  Tag directiveTagHelper(const llvm::StringRef Str) {
    return tokenTag(Str, ptml::c::tokens::Directive);
  }

public:
  // Operators.
  Tag getOperator(Operator OperatorOp) const {
    return operatorTagHelper(toString(OperatorOp));
  }

  // Constants.
  Tag getConstantTag(const llvm::StringRef Str) const {
    return ptml::PTMLBuilder::tokenTag(Str, ptml::c::tokens::Constant);
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
    return ptml::PTMLBuilder::tokenTag(Str, ptml::c::tokens::StringLiteral);
  }

  // Keywords.
  Tag getKeyword(Keyword TheKeyword) const {
    return keywordTagHelper(toString(TheKeyword));
  }

  // Scopes.
  Tag getScope(Scopes TheScope) const {
    switch (TheScope) {
    case Scopes::Scope:
      return scopeTagHelper(ptml::c::scopes::Scope);
    case Scopes::Function:
      return scopeTagHelper(ptml::c::scopes::Function);
    case Scopes::FunctionBody:
      return scopeTagHelper(ptml::c::scopes::FunctionBody);
    case Scopes::StructBody:
      return scopeTagHelper(ptml::c::scopes::StructBody);
    case Scopes::UnionBody:
      return scopeTagHelper(ptml::c::scopes::UnionBody);
    case Scopes::TypeDeclarations:
      return scopeTagHelper(ptml::c::scopes::TypeDeclarationsList);
    case Scopes::FunctionDeclarations:
      return scopeTagHelper(ptml::c::scopes::FunctionDeclarationsList);
    case Scopes::DynamicFunctionDeclarations:
      return scopeTagHelper(ptml::c::scopes::DynamicFunctionDeclarationsList);
    case Scopes::SegmentDeclarations:
      return scopeTagHelper(ptml::c::scopes::SegmentDeclarationsList);
    default:
      revng_unreachable("Unknown scope");
    }
  }

  // Directives.
  Tag getDirective(Directive TheDirective) {
    return directiveTagHelper(toString(TheDirective));
  }

  // Helpers.
  std::string getPragmaOnce() {
    return getDirective(Directive::Pragma) + " " + getConstantTag("once")
           + "\n";
  }

  std::string getIncludeAngle(const llvm::StringRef Str) {
    std::string TheStr;
    if (!isGenerateTagLessPTML())
      TheStr = "&lt;" + Str.str() + "&gt;";
    else
      TheStr = "<" + Str.str() + ">";

    return getDirective(Directive::Include) + " " + getStringLiteral(TheStr)
           + "\n";
  }

  std::string getIncludeQuote(const llvm::StringRef Str) {
    std::string TheStr;
    if (!isGenerateTagLessPTML())
      TheStr = "&quot;" + Str.str() + "&quot;";
    else
      TheStr = "\"" + Str.str() + "\"";

    return getDirective(Directive::Include) + " " + getStringLiteral(TheStr)
           + "\n";
  }

  std::string getBlockComment(const llvm::StringRef Str,
                              bool Newline = true) const {
    return ptml::PTMLBuilder::tokenTag("/* " + Str.str() + " */",
                                       ptml::tokens::Comment)
           + (Newline ? "\n" : "");
  }

  std::string getLineComment(const llvm::StringRef Str) {
    revng_check(!Str.contains('\n'));
    return ptml::PTMLBuilder::tokenTag("// " + Str.str(), ptml::tokens::Comment)
           + "\n";
  }

  std::string getAttribute(const llvm::StringRef Str) {
    return getDirective(Directive::Attribute) + "((" + Str.str() + "))";
  }

  std::string getAnnotateABI(const llvm::StringRef ABI) {
    std::string AnnotateABI = "annotate(\"abi:" + ABI.str() + "\")";
    return getAttribute(AnnotateABI);
  }

  std::string getAnnotateReg(const llvm::StringRef RegName) {
    std::string AnnotateReg = "annotate(\"reg:" + RegName.str() + "\")";
    return getAttribute(AnnotateReg);
  }

  std::string getAttributePacked() { return getAttribute("packed"); }

  Tag getNameTag(const model::Type &T) const {
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

    return ptml::PTMLBuilder::tokenTag(Name.str().str(), ptml::c::tokens::Type);
  }

  // Locations.
  constexpr const char *getLocationAttribute(bool IsDefinition) const {
    return IsDefinition ? ptml::attributes::LocationDefinition :
                          ptml::attributes::LocationReferences;
  }

  std::string serializeLocation(const model::Type &T) const {
    if (isGenerateTagLessPTML())
      return "";
    return pipeline::serializedLocation(revng::ranks::Type, T.key());
  }

  std::string getLocation(bool IsDefinition, const model::Type &T) const {
    auto Result = getNameTag(T);
    if (isGenerateTagLessPTML())
      return Result.serialize();

    Result.addAttribute(getLocationAttribute(IsDefinition),
                        serializeLocation(T));
    // non-primitive types are editable
    if (not llvm::isa<model::PrimitiveType>(&T))
      Result.addAttribute(ptml::attributes::ModelEditPath,
                          modelEditPath::getCustomNamePath(T));
    return Result.serialize();
  }

  std::string getLocationDefinition(const model::Type &T) const {
    return getLocation(true, T);
  }

  std::string getLocationReference(const model::Type &T) const {
    return getLocation(false, T);
  }

  std::string serializeLocation(const model::Segment &T) const {
    if (isGenerateTagLessPTML())
      return "";
    return pipeline::serializedLocation(revng::ranks::Segment, T.key());
  }

  Tag getNameTag(const model::Segment &S) const {
    return ptml::PTMLBuilder::tokenTag(S.name(), ptml::c::tokens::Variable);
  }

  std::string getLocation(bool IsDefinition, const model::Segment &S) const {
    return getNameTag(S)
      .addAttribute(getLocationAttribute(IsDefinition), serializeLocation(S))
      .addAttribute(ptml::attributes::ModelEditPath,
                    modelEditPath::getCustomNamePath(S))
      .serialize();
  }

  std::string getLocationDefinition(const model::Segment &S) const {
    return getLocation(true, S);
  }

  std::string getLocationReference(const model::Segment &S) const {
    return getLocation(false, S);
  }

  std::string serializeLocation(const model::EnumType &Enum,
                                const model::EnumEntry &Entry) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::serializedLocation(revng::ranks::EnumEntry,
                                        Enum.key(),
                                        Entry.key());
  }

  std::string serializeLocation(const model::StructType &Struct,
                                const model::StructField &Field) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::serializedLocation(revng::ranks::StructField,
                                        Struct.key(),
                                        Field.key());
  }

  std::string serializeLocation(const model::UnionType &Union,
                                const model::UnionField &Field) const {
    if (isGenerateTagLessPTML())
      return "";

    return pipeline::serializedLocation(revng::ranks::UnionField,
                                        Union.key(),
                                        Field.key());
  }

  Tag getNameTag(const model::EnumType &Enum,
                 const model::EnumEntry &Entry) const {
    return ptml::PTMLBuilder::tokenTag(Enum.name().str().str() + "_"
                                         + Entry.CustomName().str().str(),
                                       ptml::c::tokens::Field);
  }

  std::string getLocation(bool IsDefinition,
                          const model::EnumType &Enum,
                          const model::EnumEntry &Entry) const {
    return getNameTag(Enum, Entry)
      .addAttribute(getLocationAttribute(IsDefinition),
                    serializeLocation(Enum, Entry))
      .addAttribute(ptml::attributes::ModelEditPath,
                    modelEditPath::getCustomNamePath(Enum, Entry))
      .serialize();
  }

  template<class Field>
  Tag getNameTag(const Field &F) const {
    return ptml::PTMLBuilder::tokenTag(F.name(), c::tokens::Field);
  }

  template<typename Aggregate, typename Field>
  std::string
  getLocation(bool IsDefinition, const Aggregate &A, const Field &F) const {
    return getNameTag(F)
      .addAttribute(getLocationAttribute(IsDefinition), serializeLocation(A, F))
      .addAttribute(attributes::ModelEditPath,
                    modelEditPath::getCustomNamePath(A, F))
      .serialize();
  }

  template<typename Aggregate, typename Field>
  std::string getLocationDefinition(const Aggregate &A, const Field &F) const {
    return getLocation(true, A, F);
  }

  template<typename Aggregate, typename Field>
  std::string getLocationReference(const Aggregate &A, const Field &F) const {
    return getLocation(false, A, F);
  }
};
} // namespace ptml

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
        const llvm::StringRef Attribute = ptml::c::scopes::Scope) :
    BraceScope(Out),
    ScopeTag(Out.getPTMLBuilder()
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
  ptml::PTMLBuilder Builder;
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
