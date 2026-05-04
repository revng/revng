#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"
#include "revng/Support/CDataModel.h"

namespace ptml {

/// Provides a stream-like interface for emitting C tokens and simple
/// preprocessor directives. It is ensured that through this interface, only
/// lexically valid C code can be emitted.
class CTokenEmitter {
  // It is very important to hide the PTML emitter and not to expose any direct
  // access to it in the public interface of this class. This design prevents
  // the emission of lexically invalid C.
  PTMLStreamEmitter PTML;

  // Used to ensure that only one comment emitter may exist at any given time.
  bool IsEmittingComment = false;

  // This ensures the extra `<div></div>` we need for multi-element artifacts
  // is properly emitted (PTML requires each document to be a single tag).
  //
  // TODO: eventually we will want to introduce a separate emitter layer (think
  //       along the lines of a `DocumentEmitter`) to take care of this instead.
  PTMLTagEmitter MainTag;

public:
  explicit CTokenEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    PTML(OS, Tags), MainTag(PTML.makeTagInitializer(ptml::tags::Div)) {

    MainTag.finalizeOpenTag();
  }

  void emitSpace() { PTML.emit(" "); }
  void emitNewline() { PTML.emit("\n"); }

  enum class Keyword {
    Auto,
    Bool,
    Break,
    Case,
    Char,
    Const,
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
    Register,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    True,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,
  };

  void emitKeyword(Keyword K);

  enum class Punctuator {
    Colon,
    Comma,
    Dot,
    Equals,
    LeftBrace,
    LeftBracket,
    LeftParenthesis,
    RightBrace,
    RightBracket,
    RightParenthesis,
    Semicolon,
    Star,
  };

  void emitPunctuator(Punctuator P);

  enum class Operator {
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
    RightBracket,
    RightParenthesis,
    Slash,
    SlashEquals,
    Star,
    StarEquals,
    Tilde,
  };

  void emitOperator(Operator O);

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

    Attribute,
    AttributeArgument,

    Macro,
  };

  enum class IdentifierKind : bool {
    Reference,
    Definition,
  };

  /// \pre \param Identifier matches `[_a-zA-Z][_a-zA-Z0-9]*`.
  void emitIdentifier(llvm::StringRef Identifier,
                      llvm::StringRef Location,
                      EntityKind Kind,
                      IdentifierKind IsDefinition);

  /// \pre \param Identifier matches `[_a-zA-Z][_a-zA-Z0-9]*`.
  void emitLiteralIdentifier(llvm::StringRef Identifier);

  // TODO: There is currently no API for emitting character literals, because
  //       there are no Clift users of such an API. Whenever support for
  //       emitting character literals is needed, another function should be
  //       added for that purpose.

  struct IntegerSuffix {
    bool Unsigned;
    CStandardType MinimumType;
  };

  /// \pre \param Radix must be one of 2, 8, 10 or 16.
  void emitIntegerLiteral(llvm::APInt Value,
                          std::optional<IntegerSuffix> Suffix,
                          uint64_t Radix = 10);

  void emitStringLiteral(llvm::StringRef Content);

public:
  enum class CommentKind : bool {
    // //Looks like this
    Line,

    // /*Looks like this*/
    Block,
  };

  /// PTMLEmitter for emitting tags and content within a C line or block
  /// comment.
  class CommentEmitter {
    // A single reference to CTokenEmitter would be fine, and arguably clearer,
    // but it was decided to use two references to avoid storing a reference to
    // CTokenEmitter.
    PTMLStreamEmitter &PTML;
    bool &IsEmittingComment;

    CommentKind Kind;
    bool IsAtBeginningOfLine = false;

    std::optional<PTMLTagEmitter> Tag;

  public:
    explicit CommentEmitter(CTokenEmitter &Emitter, CommentKind Kind);

    CommentEmitter(const CommentEmitter &) = delete;
    CommentEmitter &operator=(const CommentEmitter &) = delete;

    ~CommentEmitter();

    [[nodiscard]] auto makeTagInitializer(llvm::StringRef Tag) {
      return PTML.makeTagInitializer(Tag);
    }

    [[nodiscard]] PTMLTagEmitter initializeOpenTag(llvm::StringRef Tag) {
      return PTML.initializeOpenTag(Tag);
    }

    void emit(llvm::StringRef Content);

  private:
    void emitLinePrefix();
    void emitEscaped(llvm::StringRef Content);
  };

  /// Convenience function for initializing a CommentEmitter of the specified
  /// comment kind.
  [[nodiscard]] CommentEmitter emitComment(CommentKind Kind) {
    return CommentEmitter(*this, Kind);
  }

  /// Convenience function for directly emitting a comment with the specified
  /// content.
  void emitComment(llvm::StringRef Content, CommentKind Kind);

public:
  enum class IncludeMode : bool {
    Quote,
    Angle,
  };

  void emitIncludeDirective(llvm::StringRef Content,
                            llvm::StringRef Location,
                            IncludeMode Mode);
  void emitPragmaOnceDirective();

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
    CTokenEmitter &Emitter;
    std::optional<PTMLTagEmitter> Tag;

    CTokenEmitter::Delimiter Delimiter;
    int Indent;

  public:
    explicit Scope(CTokenEmitter &Emitter,
                   ScopeKind Kind,
                   CTokenEmitter::Delimiter Delimiter,
                   int Indent);

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope();
  };

  [[nodiscard]] Scope
  enterScope(ScopeKind Kind, Delimiter Delimiter, int Indent = 1) {
    return Scope(*this, Kind, Delimiter, Indent);
  }

  enum class RegionKind : uint8_t {
    Expression,
    Commentable,
  };

  class Region {
    std::optional<PTMLTagEmitter> Tag;

  public:
    explicit Region(CTokenEmitter &Emitter,
                    RegionKind Kind,
                    llvm::StringRef Location);

    Region(const Region &) = delete;
    Region &operator=(const Region &) = delete;
  };

  [[nodiscard]] Region enterRegion(RegionKind Kind, llvm::StringRef Location) {
    return Region(*this, Kind, Location);
  }
};
static_assert(PTMLEmitter<CTokenEmitter::CommentEmitter>);

} // namespace ptml
