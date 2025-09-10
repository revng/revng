#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PTML/Emitter.h"
#include "revng/Support/CTarget.h"

class CEmitter {
public:
  explicit CEmitter(llvm::raw_ostream &OS, ptml::Tagging Tags) :
    PTML(OS, Tags) {}

  void emitSpace() { PTML.emitLiteral(" "); }

  void emitNewline() { PTML.emitNewline(); }

  enum class Keyword {
    Alignas,
    Alignof,
    Auto,
    Bool,
    Break,
    Case,
    Char,
    Const,
    Constexpr,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    False,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Nullptr,
    Register,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Static_assert,
    Struct,
    Switch,
    Thread_local,
    True,
    Typedef,
    Typeof,
    Typeof_unqual,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
  };

  void emitKeyword(Keyword K);

  enum class Symbol {
    Ampersand,
    AmpersandAmpersand,
    AmpersandEquals,
    Arrow,
    Caret,
    CaretEquals,
    Colon,
    Comma,
    Dot,
    Equals,
    EqualsEquals,
    Exclaim,
    ExclaimEquals,
    Greater,
    GreaterEquals,
    GreaterGreater,
    GreaterGreaterEquals,
    LeftBrace,
    LeftBracket,
    LeftParenthesis,
    Less,
    LessEquals,
    LessLess,
    LessLessEquals,
    Minus,
    MinusEquals,
    MinusMinus,
    Percent,
    PercentEquals,
    Pipe,
    PipeEquals,
    PipePipe,
    Plus,
    PlusEquals,
    PlusPlus,
    Question,
    RightBrace,
    RightBracket,
    RightParenthesis,
    Semicolon,
    Slash,
    SlashEquals,
    Star,
    StarEquals,
    Tilde,
  };

  void emitOperator(Symbol S);
  void emitPunctuator(Symbol S);

  enum class EntityKind {
    Primitive,
    Typedef,

    Enum,
    Enumerator,

    Struct,
    Union,
    Field,

    GlobalVariable,
    LocalVariable,

    Function,
    FunctionParameter,

    Label,
  };

  enum class IdentifierKind : bool {
    Reference,
    Definition,
  };

  /// @pre @param Identifier matches `[_a-zA-Z][_a-zA-Z0-9]*`.
  void emitIdentifier(llvm::StringRef Identifier,
                      llvm::StringRef Location,
                      EntityKind Kind,
                      IdentifierKind IsDefinition);

  /// @pre @param Identifier matches `[_a-zA-Z][_a-zA-Z0-9]*`.
  void emitLiteralIdentifier(llvm::StringRef Identifier);

  /// @pre @param Radix must be one of 2, 8, 10 or 16.
  void
  emitIntegerLiteral(llvm::APSInt Value, CIntegerKind Type, unsigned Radix);

  void emitStringLiteral(llvm::StringRef Content);

  enum class CommentKind : bool {
    Line,
    Block,
  };

  void emitComment(llvm::StringRef Content, CommentKind Kind);

  enum class IncludeMode : bool {
    Quote,
    Angle,
  };

  void emitIncludeDirective(llvm::StringRef Content,
                            llvm::StringRef Location,
                            IncludeMode Mode);

  enum class ScopeKind : uint8_t {
    None,
    EnumDefinition,
    StructDefinition,
    UnionDefinition,
    FunctionDeclaration,
    FunctionDefinition,
    BlockStatement,
  };

  enum class Delimiter : uint8_t {
    None,
    Braces,
  };

  class Scope {
  public:
    explicit Scope(CEmitter &Emitter,
                   ScopeKind Kind,
                   Delimiter Delimiter,
                   int Indent) :
      Emitter(Emitter), Delimiter(Delimiter), Indent(Indent), Tag(nullptr) {
      Emitter.enterScopeImpl(Tag, Delimiter, Indent, Kind);
    }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope() { Emitter.leaveScopeImpl(Tag, Delimiter, Indent); }

  private:
    CEmitter &Emitter;

    Delimiter Delimiter;
    int Indent;

    ptml::Emitter::TagEmitter Tag;
  };

  [[nodiscard]] Scope
  enterScope(ScopeKind Kind, Delimiter Delimiter, int Indent = 1) {
    return Scope(*this, Kind, Delimiter, Indent);
  }

private:
  void emitSymbolImpl(Symbol S, llvm::StringRef Token);

  void enterScopeImpl(ptml::Emitter::TagEmitter &Tag,
                      Delimiter Delimiter,
                      int Indent,
                      ScopeKind Kind);

  void leaveScopeImpl(ptml::Emitter::TagEmitter &Tag,
                      Delimiter Delimiter,
                      int Indent);

  ptml::Emitter PTML;
};
