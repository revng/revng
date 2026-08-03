#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"
#include "revng/Support/CDataModel.h"

namespace ptml {

/// Builder for C tokens and simple preprocessor directives.
///
/// This class ensures that through this interface, only lexically valid C code
/// can be emitted.
///
/// This class builds the output internally and reformats it
/// using `clang-format`. Users can get the output via `extract`.
// IMPORTANT: If you are about to add a new method in here, please make sure
// what it emits is wrapped into a ptml `<span>` tag with:
//
// - `data-token` set (determines syntax highlighting),
// - either `data-location-definition` or `data-location-references` set
//   (ensures correct ctrl+click behavior),
// - `data-allowed-actions` and `data-action-context-location` set iff it is
//   actionable (allows for user interactivity).
//
// Practically, you should use:
//
// - `PTML.initializeOpenTag(ptml::tags::Span)` for the tag,
// - `CTokenEmitter::getEntityKindAttribute` to select the `data-token`
//   attribute, `CTokenEmitter::getAllowedActions` to select the
//   `data-allowed-actions`.
// - `data-location-definition` or `data-location-references` should be set
//   based on the handle you already likely have.
class CTokenEmitter {
private:
  Tagging Tags;

  std::string Buffer;
  llvm::raw_string_ostream BufferStream;

  // It is very important to hide the PTML emitter and not to expose any direct
  // access to it in the public interface of this class. This design prevents
  // the emission of lexically invalid C.
  PTMLStreamEmitter PTML;

  // Used to ensure that only one comment emitter may exist at any given time.
  bool IsEmittingComment = false;

  // The extra `<div></div>` we need for multi-element artifacts (PTML requires
  // each document to be a single tag). extract() closes it before reformatting.
  //
  // TODO: eventually we will want to introduce a separate emitter layer (think
  //       along the lines of a `DocumentEmitter`) to take care of this instead.
  PTMLTagEmitter MainTag;

  // Metadata only exists to reposition whitespace in the tagged document; the
  // plain-C path reformats without it.
  static EmissionMode getEmissionMode(Tagging Tags) {
    return Tags == Tagging::Enabled ? EmissionMode::TagsAndMetadata :
                                      EmissionMode::PlainText;
  }

public:
  explicit CTokenEmitter(Tagging Tags);

public:
  // Returns the emitted document, reformatted with clang-format. Closes the
  // wrapping element, so it must be called exactly once, after all emission is
  // complete.
  [[nodiscard]] std::string extract();

  void emitSpace() { PTML.emit(" "); }
  void emitNewline() { PTML.emit("\n"); }

  enum class Keyword {
    // Standard keywords
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

    // Our custom keywords
    BreakTo,
    ContinueTo,
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
  void emitMacro(llvm::StringRef Identifier);

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

  public:
    explicit Scope(CTokenEmitter &Emitter,
                   ScopeKind Kind,
                   CTokenEmitter::Delimiter Delimiter);

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    ~Scope();
  };

  [[nodiscard]] Scope enterScope(ScopeKind Kind, Delimiter Delimiter) {
    return Scope(*this, Kind, Delimiter);
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
