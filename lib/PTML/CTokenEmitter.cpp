//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/CTokenEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Identifier.h"

// TODO: Ideally the CTokenEmitter would not have this dependency. To remove it
//       would require moving more of the logic to the UI.
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

namespace {

using EntityKind = ptml::CTokenEmitter::EntityKind;
using ScopeKind = ptml::CTokenEmitter::ScopeKind;
using Punctuator = ptml::CTokenEmitter::Punctuator;
using Operator = ptml::CTokenEmitter::Operator;

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
  default:
    revng_abort("Invalid CTokenEmitter::EntityKind");
  }
}

static llvm::StringRef getCIntegerLiteralSuffix(const CIntegerKind Type,
                                                const bool Signed) {
  switch (Type) {
  default:
  case CIntegerKind::Int:
    return Signed ? "" : "U";
  case CIntegerKind::Long:
    return Signed ? "L" : "UL";
  case CIntegerKind::LongLong:
    return Signed ? "LL" : "ULL";
  }
}

static std::optional<llvm::StringRef> getScopeKindAttribute(ScopeKind Kind) {
  switch (Kind) {
  case ScopeKind::None:
    break;
  case ScopeKind::EnumDefinition:
    return std::nullopt;
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

static llvm::SmallVector<llvm::StringRef, 2>
getAllowedActions(llvm::StringRef Location) {
  namespace rr = revng::ranks;
  namespace pa = ptml::actions;

  if (auto L = pipeline::locationFromString(rr::StructField, Location))
    return { pa::Rename };

  if (auto L = pipeline::locationFromString(rr::UnionField, Location))
    return { pa::Rename };

  if (auto L = pipeline::locationFromString(rr::EnumEntry, Location))
    return { pa::Rename };

  if (auto L = pipeline::locationFromString(rr::PrimitiveType, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::RawArgument, Location))
    return { pa::EditType };

  if (auto L = pipeline::locationFromString(rr::RawStackArguments, Location))
    return { pa::EditType };

  if (auto L = pipeline::locationFromString(rr::ArtificialStruct, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::HelperFunction, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::HelperStructType, Location))
    return {};

  if (auto L = pipeline::locationFromString(rr::HelperStructField, Location))
    return {};

  return { pa::Rename, pa::EditType };
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

  static StringEscape hex(uint8_t Value) {
    StringEscape E;
    E.Size = 4;
    E.Data[0] = '\\';
    E.Data[1] = 'x';
    E.Data[2] = getHexDigit(Value >> 4);
    E.Data[3] = getHexDigit(Value & 0x0F);
    return E;
  };

  operator llvm::StringRef() const { return llvm::StringRef(Data, Size); }

private:
  StringEscape() = default;
};

static StringEscape getStringEscape(char Character) {
  switch (Character) {
  case '\0':
    return StringEscape::single('0');
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
    return StringEscape::hex(Character);
  }
}

} // namespace

void ptml::CTokenEmitter::emitKeyword(Keyword K) {
  auto Emit = [this](llvm::StringRef String) {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
    Tag.finalizeOpenTag();

    PTML.emitLiteralContent(String);
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
  }
  revng_abort("Invalid CTokenEmitter::Keyword");
}

void ptml::CTokenEmitter::emitPunctuator(Punctuator P) {
  switch (P) {
  case Punctuator::Colon:
    return PTML.emitContent(":");
  case Punctuator::Comma:
    return PTML.emitContent(",");
  case Punctuator::Dot:
    return PTML.emitContent(".");
  case Punctuator::Equals:
    return PTML.emitContent("=");
  case Punctuator::LeftBrace:
    return PTML.emitContent("{");
  case Punctuator::LeftBracket:
    return PTML.emitContent("[");
  case Punctuator::LeftParenthesis:
    return PTML.emitContent("(");
  case Punctuator::RightBrace:
    return PTML.emitContent("}");
  case Punctuator::RightBracket:
    return PTML.emitContent("]");
  case Punctuator::RightParenthesis:
    return PTML.emitContent(")");
  case Punctuator::Semicolon:
    return PTML.emitContent(";");
  case Punctuator::Star:
    return PTML.emitContent("*");
  }
  revng_abort("Invalid CTokenEmitter::Punctuator");
}

void ptml::CTokenEmitter::emitOperator(Operator O) {
  auto Emit = [this](llvm::StringRef String) {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Operator);
    Tag.finalizeOpenTag();

    PTML.emitContent(String);
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

void ptml::CTokenEmitter::emitIdentifier(llvm::StringRef Identifier,
                                   llvm::StringRef Location,
                                   EntityKind Kind,
                                   IdentifierKind IsDefinition) {
  revng_assert(validateIdentifier(Identifier),
               "The specified identifier is not a valid C identifier.");

  auto LocationAttribute = IsDefinition == IdentifierKind::Definition ?
                             ptml::attributes::LocationDefinition :
                             ptml::attributes::LocationReferences;

  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  if (auto Attribute = getEntityKindAttribute(Kind))
    Tag.emitAttribute(ptml::attributes::Token, *Attribute);
  if (not Location.empty()) {
    Tag.emitAttribute(LocationAttribute, Location);
    Tag.emitAttribute(ptml::attributes::ActionContextLocation,
                      getActionContextLocation(Location));

    auto Actions = getAllowedActions(Location);
    if (not Actions.empty())
      Tag.emitListAttribute(ptml::attributes::AllowedActions, Actions);
  }
  Tag.finalizeOpenTag();

  PTML.emitLiteralContent(Identifier);
}

void ptml::CTokenEmitter::emitLiteralIdentifier(llvm::StringRef Identifier) {
  revng_assert(validateIdentifier(Identifier),
               "The specified identifier is not a valid C identifier.");

  PTML.emitLiteralContent(Identifier);
}

void ptml::CTokenEmitter::emitIntegerLiteral(llvm::APSInt Value,
                                       CIntegerKind Type,
                                       unsigned Radix) {
  constexpr auto IsValidRadix = [](unsigned Radix) {
    switch (Radix) {
    case 2:
    case 8:
    case 10:
    case 16:
      return true;
    default:
      return false;
    }
  };
  revng_assert(IsValidRadix(Radix),
               "The specified integer radix does not correspond to any "
               "C integer literal token.");

  llvm::SmallString<64> String;
  Value.toString(String,
                 Radix,
                 /*Signed=*/Value.isSigned(),
                 /*FormatAsCLiteral=*/true);

  String.append(getCIntegerLiteralSuffix(Type, Value.isSigned()));

  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Constant);
  Tag.finalizeOpenTag();

  PTML.emitLiteralContent(String);
}

void ptml::CTokenEmitter::emitStringLiteral(llvm::StringRef String) {
  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
  Tag.finalizeOpenTag();

  PTML.emitLiteralContent("\"");

  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [](char Character) {
      return requiresStringEscaping(Character);
    });

    PTML.emitContent(std::string_view(Begin, Pos));

    if (Pos != End)
      PTML.emitLiteralContent(getStringEscape(*Pos++));

    Begin = Pos;
  }

  PTML.emitLiteralContent("\"");
}

void ptml::CTokenEmitter::emitComment(llvm::StringRef Content,
                                      CommentKind Kind) {
  auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Comment);
  Tag.finalizeOpenTag();

  if (Kind == CommentKind::Line) {
    while (not Content.empty() and Content.back() == '\n')
      Content = Content.substr(0, Content.size() - 1);

    for (const auto &R : std::views::split(Content, '\n')) {
      PTML.emitLiteralContent("//");
      PTML.emitContent(std::string_view(R.begin(), R.end()));
      PTML.emitContentNewline();
    }
  } else {
    PTML.emitLiteralContent("/*");
    PTML.emitContent(Content);
    PTML.emitLiteralContent("*/");
  }
}

void ptml::CTokenEmitter::emitIncludeDirective(llvm::StringRef Content,
                                         llvm::StringRef Location,
                                         IncludeMode Mode) {
  // Emit include directive token:
  {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Directive);
    Tag.finalizeOpenTag();

    PTML.emitLiteralContent("#include");
  }

  PTML.emitLiteralContent(" ");

  // Emit include path:
  {
    auto Tag = PTML.initializeOpenTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
    Tag.finalizeOpenTag();

    PTML.emitContent(Mode == IncludeMode::Quote ? "\"" : "<");
    PTML.emitContent(Content);
    PTML.emitContent(Mode == IncludeMode::Quote ? "\"" : ">");
  }

  PTML.emitContentNewline();
}

void ptml::CTokenEmitter::enterScopeImpl(ptml::Emitter::TagEmitter &Tag,
                                   Delimiter Delimiter,
                                   int Indent,
                                   ScopeKind Kind) {
  if (auto Symbols = getDelimiterPunctuators(Delimiter))
    emitPunctuator(Symbols->first);

  Tag.initializeOpenTag(PTML, ptml::tags::Div);
  if (auto Attribute = getScopeKindAttribute(Kind))
    Tag.emitAttribute(ptml::attributes::Scope, *Attribute);
  Tag.finalizeOpenTag();

  PTML.indent(Indent);
}

void ptml::CTokenEmitter::leaveScopeImpl(ptml::Emitter::TagEmitter &Tag,
                                   Delimiter Delimiter,
                                   int Indent) {
  PTML.indent(-Indent);

  Tag.close();

  if (auto Symbols = getDelimiterPunctuators(Delimiter))
    emitPunctuator(Symbols->second);
}
