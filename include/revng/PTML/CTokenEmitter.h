#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PTML/PTMLEmitter.h"
#include "revng/Support/CTarget.h"

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

public:
  explicit CTokenEmitter(llvm::raw_ostream &OS, Tagging Tags) :
    PTML(OS, Tags) {}

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

  void emitPrimitive(llvm::StringRef Name,
                     IdentifierKind IsDefinition = IdentifierKind::Reference);

  void emitMacro(llvm::StringRef Name,
                 IdentifierKind IsDefinition = IdentifierKind::Reference);
  void
  emitMacroArgument(llvm::StringRef MacroName,
                    llvm::StringRef ArgumentName,
                    IdentifierKind IsDefinition = IdentifierKind::Reference);

  /// \pre \param Identifier matches `[_a-zA-Z][_a-zA-Z0-9]*`.
  void emitLiteralIdentifier(llvm::StringRef Identifier);

  // TODO: There is currently no API for emitting character literals, because
  //       there are no Clift users of such an API. Whenever support for
  //       emitting character literals is needed, another function should be
  //       added for that purpose.

  /// \pre \param Radix must be one of 2, 8, 10 or 16.
  void
  emitIntegerLiteral(llvm::APSInt Value, CIntegerKind Type, unsigned Radix);

  void emitUntypedIntegerLiteral(uint64_t Value);
  void emitUntypedHexLiteral(uint64_t Value);

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
  void emitCategoryComment(llvm::StringRef Content) {
    emitComment("\n", CommentKind::Line);
    emitComment(" " + Content.str(), CommentKind::Line);
    emitComment("\n", CommentKind::Line);
    emitNewline();
  }

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
    /// Doesn't emit anything. Is provided as a handy way of controlling
    /// indentation without showing up in the output.
    IndentOnly,

    /// Doesn't emit anything beyond the basic `<div>...</div>` pair.
    Basic,

    /// These are the same as \ref Basic except they also set
    /// `ptml::attributes::Scope` to an appropriate value
    FunctionDeclaration,

    /// The same as \ref Basic but it also emits the PTML attribute allowing
    /// this region to be folded.
    Foldable,

    /// The same as \ref Foldable except a brace pair ({}) is also emitted.
    BlockStatement,

    /// These are the same as \ref BlockStatement except they also set
    /// `ptml::attributes::Scope` to an appropriate value
    EnumDefinition,
    FunctionDefinition,
    StructDefinition,
    UnionDefinition,
  };

private:
  void emitScopeOpener(ScopeKind Kind) {
    switch (Kind) {
    case ScopeKind::IndentOnly:
    case ScopeKind::Basic:
    case ScopeKind::Foldable:
    case ScopeKind::FunctionDeclaration:
      return;

    case ScopeKind::BlockStatement:
    case ScopeKind::EnumDefinition:
    case ScopeKind::FunctionDefinition:
    case ScopeKind::StructDefinition:
    case ScopeKind::UnionDefinition:
      emitPunctuator(Punctuator::LeftBrace);
      return;

    default:
      revng_abort("Unknown scope kind");
    }
  }

  void emitScopeCloser(ScopeKind Kind) {
    switch (Kind) {
    case ScopeKind::IndentOnly:
    case ScopeKind::Basic:
    case ScopeKind::Foldable:
    case ScopeKind::FunctionDeclaration:
      return;

    case ScopeKind::BlockStatement:
    case ScopeKind::EnumDefinition:
    case ScopeKind::FunctionDefinition:
    case ScopeKind::StructDefinition:
    case ScopeKind::UnionDefinition:
      emitPunctuator(Punctuator::RightBrace);
      return;

    default:
      revng_abort("Unknown scope kind");
    }
  }

  void indent(int64_t LevelDifference) {
    PTML.indent(LevelDifference * TabWidth);
  }

private:
  // TODO: consider exposing this as a configuration option!
  static constexpr uint64_t TabWidth = 2;

public:
  class Scope {
    CTokenEmitter &Emitter;
    std::optional<PTMLTagEmitter> Tag;
    ScopeKind Kind;
    int Indent;

  public:
    explicit Scope(CTokenEmitter &Emitter, ScopeKind Kind, int Indent);

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope();
  };

  [[nodiscard]] Scope enterScope(ScopeKind Kind, int Indent = 1) {
    return Scope(*this, Kind, Indent);
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
