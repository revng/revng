//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>
#include <string>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/LineRange.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Identifier.h"

// TODO: Ideally the CTokenEmitter would not have this dependency. To remove it
//       would require moving more of the logic to the UI.
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

using ptml::CTokenEmitter;

namespace {

using EntityKind = CTokenEmitter::EntityKind;
using ScopeKind = CTokenEmitter::ScopeKind;
using Punctuator = CTokenEmitter::Punctuator;
using Operator = CTokenEmitter::Operator;
using RegionKind = CTokenEmitter::RegionKind;

static std::optional<llvm::StringRef> getEntityKindAttribute(EntityKind Kind) {
  switch (Kind) {
  case EntityKind::Primitive:
    return ptml::c::tokens::Type;
  case EntityKind::Typedef:
    return ptml::c::tokens::Type;
  case EntityKind::Enum:
    return ptml::c::tokens::Type;
  case EntityKind::Enumerator:
    return ptml::c::tokens::Constant;
  case EntityKind::Struct:
    return ptml::c::tokens::Type;
  case EntityKind::Union:
    return ptml::c::tokens::Type;
  case EntityKind::Field:
    return ptml::c::tokens::Field;
  case EntityKind::GlobalVariable:
    return ptml::c::tokens::Variable;
  case EntityKind::LocalVariable:
    return ptml::c::tokens::Variable;
  case EntityKind::Function:
    return ptml::c::tokens::Function;
  case EntityKind::FunctionParameter:
    return ptml::c::tokens::FunctionParameter;
  case EntityKind::Label:
    return ptml::c::tokens::GotoLabel;
  case EntityKind::Attribute:
  case EntityKind::AttributeArgument:
    return std::nullopt;
  case EntityKind::Macro:
    return ptml::c::tokens::Macro;
  default:
    revng_abort("Invalid CTokenEmitter::EntityKind");
  }
}

static llvm::StringRef
getCIntegerLiteralSuffix(const ptml::CTokenEmitter::IntegerSuffix &Suffix) {
  switch (Suffix.MinimumType) {
  default:
  case CStandardType::Int:
    return Suffix.Unsigned ? "U" : "";
  case CStandardType::Long:
    return Suffix.Unsigned ? "UL" : "L";
  case CStandardType::LongLong:
    return Suffix.Unsigned ? "ULL" : "LL";
  }
}

static std::optional<llvm::StringRef> getScopeKindAttribute(ScopeKind Kind) {
  switch (Kind) {
  case ScopeKind::None:
    break;
  case ScopeKind::EnumDefinition:
    return ptml::c::scopes::EnumBody;
  case ScopeKind::StructDefinition:
    return ptml::c::scopes::StructBody;
  case ScopeKind::UnionDefinition:
    return ptml::c::scopes::UnionBody;
  case ScopeKind::FunctionDeclaration:
    return ptml::c::scopes::Function;
  case ScopeKind::FunctionDefinition:
    return ptml::c::scopes::FunctionBody;
  case ScopeKind::BlockStatement:
    return ptml::c::scopes::Scope;
  }
  return std::nullopt;
}

// Strictly speaking, this is a violation of the design (it's intended for
// the backend to not be aware of locations), but we have to do that because
// the alternative would have been to put all the allowed actions in clift
// directly from the clifter, which would make clift huge and ugly to debug.
//
// IMPORTANT: this is an explicit one-off violation of the design for specific
//            trade offs. Do not apply similar patterns elsewhere without
//            thinking it through.
//
// This function *intentionally* emits "no allowed actions" for the locations
// we don't know. Which in practice means that the backend can work perfectly
// fine in PTML mode even with garbage locations.
static llvm::SmallVector<llvm::StringRef, 2>
getAllowedActions(llvm::StringRef Location) {
  namespace rr = revng::ranks;
  namespace pa = ptml::actions;

  if (auto L = pipeline::locationFromString(rr::TypeDefinition, Location))
    return { pa::Rename, pa::EditType };
  if (auto L = pipeline::locationFromString(rr::Function, Location))
    return { pa::Rename, pa::EditType };
  if (auto L = pipeline::locationFromString(rr::DynamicFunction, Location))
    return { pa::Rename, pa::EditType };
  if (auto L = pipeline::locationFromString(rr::Segment, Location))
    return { pa::Rename, pa::EditType };

  if (auto L = pipeline::locationFromString(rr::StructField, Location))
    return { pa::Rename };
  if (auto L = pipeline::locationFromString(rr::UnionField, Location))
    return { pa::Rename };
  if (auto L = pipeline::locationFromString(rr::EnumEntry, Location))
    return { pa::Rename };

  if (auto L = pipeline::locationFromString(rr::PrimitiveType, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::CABIArgument, Location))
    return { pa::Rename, pa::EditType };

  if (auto L = pipeline::locationFromString(rr::RawArgument, Location))
    return { pa::EditType };
  if (auto L = pipeline::locationFromString(rr::RawStackArguments, Location))
    return { pa::EditType };
  if (auto L = pipeline::locationFromString(rr::ReturnRegister, Location))
    return { pa::Rename, pa::EditType };

  if (auto L = pipeline::locationFromString(rr::ArtificialStruct, Location))
    return {};
  if (auto L = pipeline::locationFromString(rr::HelperFunction, Location))
    return {};
  if (auto L = pipeline::locationFromString(rr::HelperStructType, Location))
    return {};
  if (auto L = pipeline::locationFromString(rr::HelperStructField, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::Macro, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::Instruction, Location))
    return { pa::CodeSwitch, pa::Comment };

  if (auto L = pipeline::locationFromString(rr::GotoLabel, Location))
    return { pa::Rename };
  if (auto L = pipeline::locationFromString(rr::LocalVariable, Location))
    return { pa::Rename, pa::EditType };
  if (auto L = pipeline::locationFromString(rr::StackFrameVariable, Location))
    return { pa::Rename, pa::EditType };

  if (auto L = pipeline::locationFromString(rr::OpaqueType, Location))
    return {};

  // TODO: we should definitely error out here. But asserting is awkward.
  //
  // Once we have a proper error handling mechanism, we should use it.
  // revng_abort(("Unknown Location: " + Location.str()).c_str());
  return {};
}

static std::string getActionContextLocation(llvm::StringRef Location) {
  namespace rr = revng::ranks;

  if (auto L = pipeline::locationFromString(rr::RawStackArguments, Location))
    return L->transmute(rr::TypeDefinition).toString();

  return Location.str();
}

static std::optional<std::pair<Punctuator, Punctuator>>
getDelimiterPunctuators(CTokenEmitter::Delimiter Delimiter) {
  switch (Delimiter) {
  case CTokenEmitter::Delimiter::None:
    break;
  case CTokenEmitter::Delimiter::Braces:
    return std::pair<Punctuator, Punctuator>(Punctuator::LeftBrace,
                                             Punctuator::RightBrace);
  }
  return std::nullopt;
}

static bool requiresStringEscaping(char Character) {
  switch (Character) {
  case '\\':
  case '\"':
  case 0x7F:
    return true;

  default:
    // Match ASCII control [0, 31] and non-ASCII characters [0x80, 0xFF]:
    return static_cast<signed char>(Character) < 32;
  }
}

static bool isOctDigit(uint8_t Value) {
  return '0' <= Value and Value <= '7';
}

static bool isHexDigit(uint8_t Value) {
  if ('0' <= Value and Value <= '7')
    return true;
  if ('a' <= Value and Value <= 'f')
    return true;
  if ('A' <= Value and Value <= 'F')
    return true;

  return false;
}

static char getHexDigit(uint8_t Value) {
  revng_assert(Value < 0x10);

  return Value < 10 ? static_cast<uint8_t>('0' + Value) :
                      static_cast<uint8_t>(('a' - 10) + Value);
}

class StringEscape {
  uint8_t Size;
  char Data[4];

public:
  static StringEscape single(char Character) {
    StringEscape E;
    E.Size = 2;
    E.Data[0] = '\\';
    E.Data[1] = Character;
    return E;
  };

  static StringEscape numeric(uint8_t Value, uint8_t NextValue) {
    if (Value < 8) {
      if (!isOctDigit(NextValue))
        return oneDigitOctal(Value);
    } else {
      if (!isHexDigit(NextValue))
        return twoDigitHex(Value);
    }
    return threeDigitOctal(Value);
  }

  static StringEscape twoDigitHex(uint8_t Value) {
    StringEscape E;
    E.Size = 4;
    E.Data[0] = '\\';
    E.Data[1] = 'x';
    E.Data[2] = getHexDigit(Value >> 4);
    E.Data[3] = getHexDigit(Value & 0x0F);
    return E;
  };

  static StringEscape oneDigitOctal(uint8_t Value) {
    revng_assert(Value < 8);
    StringEscape E;
    E.Size = 2;
    E.Data[0] = '\\';
    E.Data[1] = '0' + Value;
    return E;
  };

  static StringEscape threeDigitOctal(uint8_t Value) {
    StringEscape E;
    E.Size = 4;
    E.Data[0] = '\\';
    E.Data[1] = '0' + (Value >> 6 & 0b111);
    E.Data[2] = '0' + (Value >> 3 & 0b111);
    E.Data[3] = '0' + (Value >> 0 & 0b111);
    return E;
  };

  operator llvm::StringRef() const { return llvm::StringRef(Data, Size); }

private:
  StringEscape() = default;
};

static StringEscape getStringEscape(char Character, char NextCharacter) {
  switch (Character) {
  case '\t':
    return StringEscape::single('t');
  case '\n':
    return StringEscape::single('n');
  case '\v':
    return StringEscape::single('v');
  case '\f':
    return StringEscape::single('f');
  case '\r':
    return StringEscape::single('r');
  case '\\':
    return StringEscape::single('\\');
  case '\"':
    return StringEscape::single('\"');
  default:
    return StringEscape::numeric(Character, NextCharacter);
  }
}

} // namespace

//===---------------------------- CTokenEmitter ---------------------------===//

void CTokenEmitter::emitKeyword(Keyword K) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  auto Emit = [this](llvm::StringRef String) {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
    Tag.finalizeOpenTag();

    PTML.emit(String);
  };

  switch (K) {
  case Keyword::Auto:
    return Emit("auto");
  case Keyword::Bool:
    return Emit("bool");
  case Keyword::Break:
    return Emit("break");
  case Keyword::Case:
    return Emit("case");
  case Keyword::Char:
    return Emit("char");
  case Keyword::Const:
    return Emit("const");
  case Keyword::Continue:
    return Emit("continue");
  case Keyword::Default:
    return Emit("default");
  case Keyword::Do:
    return Emit("do");
  case Keyword::Double:
    return Emit("double");
  case Keyword::Else:
    return Emit("else");
  case Keyword::Enum:
    return Emit("enum");
  case Keyword::Extern:
    return Emit("extern");
  case Keyword::False:
    return Emit("false");
  case Keyword::For:
    return Emit("for");
  case Keyword::Goto:
    return Emit("goto");
  case Keyword::If:
    return Emit("if");
  case Keyword::Inline:
    return Emit("inline");
  case Keyword::Int:
    return Emit("int");
  case Keyword::Long:
    return Emit("long");
  case Keyword::Register:
    return Emit("register");
  case Keyword::Return:
    return Emit("return");
  case Keyword::Short:
    return Emit("short");
  case Keyword::Signed:
    return Emit("signed");
  case Keyword::Sizeof:
    return Emit("sizeof");
  case Keyword::Static:
    return Emit("static");
  case Keyword::Struct:
    return Emit("struct");
  case Keyword::Switch:
    return Emit("switch");
  case Keyword::True:
    return Emit("true");
  case Keyword::Typedef:
    return Emit("typedef");
  case Keyword::Union:
    return Emit("union");
  case Keyword::Unsigned:
    return Emit("unsigned");
  case Keyword::Void:
    return Emit("void");
  case Keyword::Volatile:
    return Emit("volatile");
  case Keyword::While:
    return Emit("while");
  case Keyword::BreakTo:
    return Emit("break_to");
  case Keyword::ContinueTo:
    return Emit("continue_to");
  }
  revng_abort("Invalid CTokenEmitter::Keyword");
}

void CTokenEmitter::emitPunctuator(Punctuator P) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  auto Emit = [this](llvm::StringRef String) {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Punctuation);
    Tag.finalizeOpenTag();

    PTML.emit(String);
  };

  switch (P) {
  case Punctuator::Colon:
    return Emit(":");
  case Punctuator::Comma:
    return Emit(",");
  case Punctuator::Dot:
    return Emit(".");
  case Punctuator::Equals:
    return Emit("=");
  case Punctuator::LeftBrace:
    return Emit("{");
  case Punctuator::LeftBracket:
    return Emit("[");
  case Punctuator::LeftParenthesis:
    return Emit("(");
  case Punctuator::RightBrace:
    return Emit("}");
  case Punctuator::RightBracket:
    return Emit("]");
  case Punctuator::RightParenthesis:
    return Emit(")");
  case Punctuator::Semicolon:
    return Emit(";");
  case Punctuator::Star:
    return Emit("*");
  }
  revng_abort("Invalid CTokenEmitter::Punctuator");
}

void CTokenEmitter::emitOperator(Operator O) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  auto Emit = [this](llvm::StringRef String) {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Operator);
    Tag.finalizeOpenTag();

    PTML.emit(String);
  };

  switch (O) {
  case Operator::Ampersand:
    return Emit("&");
  case Operator::AmpersandAmpersand:
    return Emit("&&");
  case Operator::AmpersandEquals:
    return Emit("&=");
  case Operator::Arrow:
    return Emit("->");
  case Operator::Caret:
    return Emit("^");
  case Operator::CaretEquals:
    return Emit("^=");
  case Operator::Colon:
    return Emit(":");
  case Operator::Comma:
    return Emit(",");
  case Operator::Dot:
    return Emit(".");
  case Operator::Equals:
    return Emit("=");
  case Operator::EqualsEquals:
    return Emit("==");
  case Operator::Exclaim:
    return Emit("!");
  case Operator::ExclaimEquals:
    return Emit("!=");
  case Operator::Greater:
    return Emit(">");
  case Operator::GreaterEquals:
    return Emit(">=");
  case Operator::GreaterGreater:
    return Emit(">>");
  case Operator::GreaterGreaterEquals:
    return Emit(">>=");
  case Operator::LeftBracket:
    return Emit("[");
  case Operator::LeftParenthesis:
    return Emit("(");
  case Operator::Less:
    return Emit("<");
  case Operator::LessEquals:
    return Emit("<=");
  case Operator::LessLess:
    return Emit("<<");
  case Operator::LessLessEquals:
    return Emit("<<=");
  case Operator::Minus:
    return Emit("-");
  case Operator::MinusEquals:
    return Emit("-=");
  case Operator::MinusMinus:
    return Emit("--");
  case Operator::Percent:
    return Emit("%");
  case Operator::PercentEquals:
    return Emit("%=");
  case Operator::Pipe:
    return Emit("|");
  case Operator::PipeEquals:
    return Emit("|=");
  case Operator::PipePipe:
    return Emit("||");
  case Operator::Plus:
    return Emit("+");
  case Operator::PlusEquals:
    return Emit("+=");
  case Operator::PlusPlus:
    return Emit("++");
  case Operator::Question:
    return Emit("?");
  case Operator::RightBracket:
    return Emit("]");
  case Operator::RightParenthesis:
    return Emit(")");
  case Operator::Slash:
    return Emit("/");
  case Operator::SlashEquals:
    return Emit("/=");
  case Operator::Star:
    return Emit("*");
  case Operator::StarEquals:
    return Emit("*=");
  case Operator::Tilde:
    return Emit("~");
  }
  revng_abort("Invalid CTokenEmitter::Operator");
}

void CTokenEmitter::emitIdentifier(llvm::StringRef Identifier,
                                   llvm::StringRef Location,
                                   EntityKind Kind,
                                   IdentifierKind IsDefinition) {
  if (IsEmittingComment && IsDefinition == IdentifierKind::Definition) {
    std::string Error = "An identifier ('" + Identifier.str()
                        + "') can only be referenced in a comment. Definitions "
                          "are not allowed.";
    revng_abort(Error.c_str());
  }

  revng_check(!Identifier.empty());
  if (not validateIdentifier(Identifier)) {
    revng_abort(("`" + Identifier + "` is not a valid C identifier.")
                  .str()
                  .c_str());
  }

  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  if (auto Attribute = getEntityKindAttribute(Kind))
    Tag.emitAttribute(ptml::attributes::Token, *Attribute);

  if (not Location.empty()) {
    auto LocationAttribute = IsDefinition == IdentifierKind::Definition ?
                               ptml::attributes::LocationDefinition :
                               ptml::attributes::LocationReferences;
    Tag.emitAttribute(LocationAttribute, Location);

    auto Actions = getAllowedActions(Location);
    if (not Actions.empty()) {
      Tag.emitAttribute(ptml::attributes::ActionContextLocation,
                        getActionContextLocation(Location));
      Tag.emitListAttribute(ptml::attributes::AllowedActions, Actions);
    }
  }
  Tag.finalizeOpenTag();

  PTML.emit(Identifier);
}

void CTokenEmitter::emitMacro(llvm::StringRef Identifier) {
  auto Location = pipeline::locationString(revng::ranks::Macro,
                                           Identifier.str());
  emitIdentifier(Identifier,
                 Location,
                 EntityKind::Macro,
                 IdentifierKind::Reference);
}

static bool isRadixSupported(uint64_t Radix) {
  switch (Radix) {
  case 2:
  case 8:
  case 10:
  case 16:
    return true;

  default:
    return false;
  };
}

void CTokenEmitter::emitIntegerLiteral(llvm::APInt Value,
                                       std::optional<IntegerSuffix> Suffix,
                                       uint64_t Radix) {
  revng_assert(not Suffix or isIntegerType(Suffix->MinimumType));
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  if (not isRadixSupported(Radix)) {
    std::string Error = "The specified integer radix (" + std::to_string(Radix)
                        + ") does not correspond to any C integer literal "
                          "token.";
    revng_abort(Error.c_str());
  }

  llvm::SmallString<64> String;
  Value.toString(String,
                 Radix,
                 /* Signed = */ false,
                 /* FormatAsCLiteral = */ true);

  if (Suffix.has_value())
    String.append(getCIntegerLiteralSuffix(Suffix.value()));

  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Constant);
  Tag.finalizeOpenTag();

  PTML.emit(String);
}

void CTokenEmitter::emitStringLiteral(llvm::StringRef String) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
  Tag.finalizeOpenTag();

  PTML.emit("\"");

  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [](char Character) {
      return requiresStringEscaping(Character);
    });

    PTML.emit(std::string_view(Begin, Pos));

    if (Pos != End) {
      char ThisChar = *Pos++;
      char NextChar = Pos != End ? *Pos : '\0';
      PTML.emit(getStringEscape(ThisChar, NextChar));
    }

    Begin = Pos;
  }

  PTML.emit("\"");
}

void CTokenEmitter::emitComment(llvm::StringRef Content, CommentKind Kind) {
  emitComment(Kind).emit(Content);
}

static void emitDirectiveIdentifier(ptml::PTMLStreamEmitter &PTML,
                                    llvm::StringRef Name) {
  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Directive);
  Tag.finalizeOpenTag();

  PTML.emit("#");

  revng_assert(validateIdentifier(Name));
  PTML.emit(Name);
}

void CTokenEmitter::emitIncludeDirective(llvm::StringRef Content,
                                         llvm::StringRef Location,
                                         IncludeMode Mode) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  emitDirectiveIdentifier(PTML, "include");
  emitSpace();

  // Emit include path:
  {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
    Tag.finalizeOpenTag();

    PTML.emit(Mode == IncludeMode::Quote ? "\"" : "<");
    PTML.emit(Content);
    PTML.emit(Mode == IncludeMode::Quote ? "\"" : ">");
  }

  PTML.emit("\n");
}

void ptml::CTokenEmitter::emitPragmaOnceDirective() {
  emitDirectiveIdentifier(PTML, "pragma");
  emitSpace();

  {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Constant);
    Tag.finalizeOpenTag();

    PTML.emit("once");
  }

  emitNewline();
}

CTokenEmitter::Scope::Scope(CTokenEmitter &Emitter,
                            ScopeKind Kind,
                            CTokenEmitter::Delimiter Delimiter,
                            int Indent) :
  Emitter(Emitter), Delimiter(Delimiter), Indent(Indent) {
  revng_assert(not Emitter.IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  if (auto Symbols = getDelimiterPunctuators(Delimiter))
    Emitter.emitPunctuator(Symbols->first);

  if (auto Attribute = getScopeKindAttribute(Kind)) {
    Tag.emplace(Emitter.PTML.makeTagInitializer(ptml::tags::Div));
    Tag->emitAttribute(ptml::attributes::Scope, *Attribute);
    Tag->finalizeOpenTag();
  }

  Emitter.PTML.indent(Indent);
}

CTokenEmitter::Scope::~Scope() {
  Emitter.PTML.indent(-Indent);

  Tag.reset();

  if (auto Symbols = getDelimiterPunctuators(Delimiter))
    Emitter.emitPunctuator(Symbols->second);
}

CTokenEmitter::Region::Region(CTokenEmitter &Emitter,
                              RegionKind Kind,
                              llvm::StringRef Location) {
  if (Location.empty())
    return;

  llvm::SmallVector<llvm::StringRef, 2> Actions = {};
  switch (Kind) {
  case RegionKind::Expression:
    Actions = getAllowedActions(Location);
    break;

  case RegionKind::Commentable:
    Actions = { ptml::actions::Comment };
    break;

  default:
    revng_abort("Unknown region kind");
  };

  if (Actions.empty())
    return;

  Tag.emplace(Emitter.PTML.makeTagInitializer(ptml::tags::Span));
  Tag->emitAttribute(ptml::attributes::ActionContextLocation,
                     getActionContextLocation(Location));
  Tag->emitListAttribute(ptml::attributes::AllowedActions, Actions);
  Tag->finalizeOpenTag();
}

//===-------------------- CTokenEmitter::CommentEmitter -------------------===//

CTokenEmitter::CommentEmitter::CommentEmitter(CTokenEmitter &Emitter,
                                              CommentKind Kind) :
  // NOTE: IsEmittingComment is a reference to a boolean.
  PTML(Emitter.PTML), IsEmittingComment(Emitter.IsEmittingComment), Kind(Kind) {
  revng_assert(not IsEmittingComment,
               "Cannot emit tokens while an open CommentEmitter exists.");

  revng_assert(not IsEmittingComment);
  IsEmittingComment = true;

  Tag.emplace(PTML.makeTagInitializer(ptml::tags::Span));
  Tag->emitAttribute(ptml::attributes::Token, ptml::tokens::Comment);
  Tag->finalizeOpenTag();

  switch (Kind) {
  case CommentKind::Line:
    PTML.emit("//");
    break;

  case CommentKind::Block:
    PTML.emit("/*");
    break;
  }
}

CTokenEmitter::CommentEmitter::~CommentEmitter() {
  switch (Kind) {
  case CommentKind::Line:
    if (not IsAtBeginningOfLine)
      PTML.emit("\n");
    break;

  case CommentKind::Block:
    PTML.emit("*/");
    break;
  }

  Tag.reset();

  IsEmittingComment = false;
}

void CTokenEmitter::CommentEmitter::emit(llvm::StringRef Content) {
  if (not Content.empty()) {
    bool EmitLinePrefix = IsAtBeginningOfLine;

    for (auto Line : LineRange(Content)) {
      if (std::exchange(EmitLinePrefix, true))
        emitLinePrefix();

      if (Line == "\n")
        PTML.emit("\n");
      else
        emitEscaped(Line);
    }

    IsAtBeginningOfLine = Content.back() == '\n';
  }
}

void CTokenEmitter::CommentEmitter::emitLinePrefix() {
  switch (Kind) {
  case CommentKind::Line:
    PTML.emit("//");
    break;

  case CommentKind::Block:
    break;
  }
}

void CTokenEmitter::CommentEmitter::emitEscaped(llvm::StringRef Content) {
  revng_assert(not Content.empty());

  switch (Kind) {
  case CommentKind::Line:
    PTML.emit(Content);
    break;

  case CommentKind::Block:
    for (auto [I, R] : llvm::enumerate(std::views::split(Content, "*/"))) {
      if (I != 0)
        PTML.emit("  ");

      PTML.emit(std::string_view(R.begin(), R.end()));
    }
    break;
  }
}
