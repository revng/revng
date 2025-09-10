//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PTML/CEmitter.h"
#include "revng/PTML/Constants.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Identifier.h"

// TODO: Ideally the CEmitter would not have this dependency. To remove it would
//       require moving more of the logic to the UI.
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"

namespace rr = revng::ranks;

namespace {

using EntityKind = CEmitter::EntityKind;
using ScopeKind = CEmitter::ScopeKind;
using Symbol = CEmitter::Symbol;

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
    revng_abort("Invalid CEmitter::EntityKind");
  }
}

static llvm::StringRef getCIntegerLiteralSuffix(const CIntegerKind Type,
                                                const bool Signed) {
  switch (Type) {
  default:
  case CIntegerKind::Int:
    return Signed ? "" : "u";
  case CIntegerKind::Long:
    return Signed ? "l" : "ul";
  case CIntegerKind::LongLong:
    return Signed ? "ll" : "ull";
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

static llvm::ArrayRef<llvm::StringRef>
getAllowedActions(llvm::StringRef Location) {
  auto GetActions = [&](bool Rename, bool EditType) {
    static constexpr llvm::StringRef Actions[] = {
      ptml::actions::Rename,
      ptml::actions::EditType,
    };

    return llvm::ArrayRef(Actions).slice(!Rename, Rename + EditType);
  };

  if (auto L = pipeline::locationFromString(rr::StructField, Location))
    return GetActions(true, false);

  if (auto L = pipeline::locationFromString(rr::UnionField, Location))
    return GetActions(true, false);

  if (auto L = pipeline::locationFromString(rr::EnumEntry, Location))
    return GetActions(true, false);

  if (auto L = pipeline::locationFromString(rr::PrimitiveType, Location))
    return GetActions(false, false);

  if (auto L = pipeline::locationFromString(rr::RawArgument, Location))
    return GetActions(false, true);

  if (auto L = pipeline::locationFromString(rr::RawStackArguments, Location))
    return GetActions(false, true);

  if (auto L = pipeline::locationFromString(rr::ArtificialStruct, Location))
    return GetActions(false, false);

  if (auto L = pipeline::locationFromString(rr::HelperFunction, Location))
    return GetActions(false, false);

  if (auto L = pipeline::locationFromString(rr::HelperStructType, Location))
    return GetActions(false, false);

  if (auto L = pipeline::locationFromString(rr::HelperStructField, Location))
    return GetActions(false, false);

  return GetActions(true, true);
}

static std::string getActionContextLocation(llvm::StringRef Location) {
  if (auto L = pipeline::locationFromString(rr::RawStackArguments, Location))
    return L->transmute(rr::TypeDefinition).toString();

  return Location.str();
}

static std::optional<std::pair<Symbol, Symbol>>
getDelimiterSymbols(CEmitter::Delimiter Delimiter) {
  switch (Delimiter) {
  case CEmitter::Delimiter::None:
    break;
  case CEmitter::Delimiter::Braces:
    return std::pair<Symbol, Symbol>(Symbol::LeftBrace, Symbol::RightBrace);
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

void CEmitter::emitKeyword(Keyword K) {
  auto Emit = [&](llvm::StringRef String) {
    auto Tag = PTML.enterTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Keyword);
    Tag.finalize();

    PTML.emitLiteral(String);
  };

  switch (K) {
  case Keyword::Alignas:
    return Emit("alignas");
  case Keyword::Alignof:
    return Emit("alignof");
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
  case Keyword::Constexpr:
    return Emit("constexpr");
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
  case Keyword::Nullptr:
    return Emit("nullptr");
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
  case Keyword::Static_assert:
    return Emit("static_assert");
  case Keyword::Struct:
    return Emit("struct");
  case Keyword::Switch:
    return Emit("switch");
  case Keyword::Thread_local:
    return Emit("thread_local");
  case Keyword::True:
    return Emit("true");
  case Keyword::Typedef:
    return Emit("typedef");
  case Keyword::Typeof:
    return Emit("typeof");
  case Keyword::Typeof_unqual:
    return Emit("typeof_unqual");
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
  revng_abort("Invalid CEmitter::Keyword");
}

void CEmitter::emitSymbolImpl(Symbol S, llvm::StringRef Token) {
  auto Emit = [&](llvm::StringRef String) {
    ptml::Emitter::TagEmitter Tag(nullptr);

    if (not Token.empty()) {
      Tag.initialize(PTML, ptml::tags::Span);
      Tag.emitAttribute(ptml::attributes::Token, Token);
      Tag.finalize();
    }

    PTML.emit(String);
  };

  switch (S) {
  case Symbol::Ampersand:
    return Emit("&");
  case Symbol::AmpersandAmpersand:
    return Emit("&&");
  case Symbol::AmpersandEquals:
    return Emit("&=");
  case Symbol::Arrow:
    return Emit("->");
  case Symbol::Caret:
    return Emit("^");
  case Symbol::CaretEquals:
    return Emit("^=");
  case Symbol::Colon:
    return Emit(":");
  case Symbol::Comma:
    return Emit(",");
  case Symbol::Dot:
    return Emit(".");
  case Symbol::Equals:
    return Emit("=");
  case Symbol::EqualsEquals:
    return Emit("==");
  case Symbol::Exclaim:
    return Emit("!");
  case Symbol::ExclaimEquals:
    return Emit("!=");
  case Symbol::Greater:
    return Emit(">");
  case Symbol::GreaterEquals:
    return Emit(">=");
  case Symbol::GreaterGreater:
    return Emit(">>");
  case Symbol::GreaterGreaterEquals:
    return Emit(">>=");
  case Symbol::LeftBrace:
    return Emit("{");
  case Symbol::LeftBracket:
    return Emit("[");
  case Symbol::LeftParenthesis:
    return Emit("(");
  case Symbol::Less:
    return Emit("<");
  case Symbol::LessEquals:
    return Emit("<=");
  case Symbol::LessLess:
    return Emit("<<");
  case Symbol::LessLessEquals:
    return Emit("<<=");
  case Symbol::Minus:
    return Emit("-");
  case Symbol::MinusEquals:
    return Emit("-=");
  case Symbol::MinusMinus:
    return Emit("--");
  case Symbol::Percent:
    return Emit("%");
  case Symbol::PercentEquals:
    return Emit("%=");
  case Symbol::Pipe:
    return Emit("|");
  case Symbol::PipeEquals:
    return Emit("|=");
  case Symbol::PipePipe:
    return Emit("||");
  case Symbol::Plus:
    return Emit("+");
  case Symbol::PlusEquals:
    return Emit("+=");
  case Symbol::PlusPlus:
    return Emit("++");
  case Symbol::Question:
    return Emit("?");
  case Symbol::RightBrace:
    return Emit("}");
  case Symbol::RightBracket:
    return Emit("]");
  case Symbol::RightParenthesis:
    return Emit(")");
  case Symbol::Semicolon:
    return Emit(";");
  case Symbol::Slash:
    return Emit("/");
  case Symbol::SlashEquals:
    return Emit("/=");
  case Symbol::Star:
    return Emit("*");
  case Symbol::StarEquals:
    return Emit("*=");
  case Symbol::Tilde:
    return Emit("~");
  }
  revng_abort("Invalid CEmitter::Symbol");
}

void CEmitter::emitOperator(Symbol S) {
  emitSymbolImpl(S, ptml::c::tokens::Operator);
}

void CEmitter::emitPunctuator(Symbol S) {
  emitSymbolImpl(S, "");
}

void CEmitter::emitIdentifier(llvm::StringRef Identifier,
                              llvm::StringRef Location,
                              EntityKind Kind,
                              IdentifierKind IsDefinition) {
  revng_assert(validateIdentifier(Identifier));

  auto LocationAttribute = IsDefinition == IdentifierKind::Definition ?
                             ptml::attributes::LocationDefinition :
                             ptml::attributes::LocationReferences;

  auto Tag = PTML.enterTag(ptml::tags::Span);
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
  Tag.finalize();

  PTML.emitLiteral(Identifier);
}

void CEmitter::emitLiteralIdentifier(llvm::StringRef Identifier) {
  revng_assert(validateIdentifier(Identifier));
  PTML.emitLiteral(Identifier);
}

void CEmitter::emitIntegerLiteral(llvm::APSInt Value,
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
  revng_assert(IsValidRadix(Radix));

  llvm::SmallString<64> String;
  Value.toString(String,
                 Radix,
                 /*Signed=*/Value.isSigned(),
                 /*FormatAsCLiteral=*/true);

  String.append(getCIntegerLiteralSuffix(Type, Value.isSigned()));

  auto Tag = PTML.enterTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Constant);
  Tag.finalize();

  PTML.emitLiteral(String);
}

void CEmitter::emitStringLiteral(llvm::StringRef String) {
  auto Tag = PTML.enterTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
  Tag.finalize();

  PTML.emitLiteral("\"");

  auto Beg = String.data();
  auto End = Beg + String.size();

  while (Beg != End) {
    auto Pos = std::find_if(Beg, End, [](char Character) {
      return requiresStringEscaping(Character);
    });

    PTML.emit(std::string_view(Beg, Pos));

    if (Pos != End)
      PTML.emitLiteral(getStringEscape(*Pos++));

    Beg = Pos;
  }

  PTML.emitLiteral("\"");
}

void CEmitter::emitComment(llvm::StringRef Content, CommentKind Kind) {
  auto Tag = PTML.enterTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Comment);
  Tag.finalize();

  if (Kind == CommentKind::Line) {
    while (not Content.empty() and Content.back() == '\n')
      Content = Content.substr(0, Content.size() - 1);

    for (const auto &R : std::views::split(Content, '\n')) {
      PTML.emitLiteral("//");
      PTML.emit(std::string_view(R.begin(), R.end()));
      PTML.emitNewline();
    }
  } else {
    PTML.emitLiteral("/*");
    PTML.emit(Content);
    PTML.emitLiteral("*/");
  }
}

void CEmitter::emitIncludeDirective(llvm::StringRef Content,
                                    llvm::StringRef Location,
                                    IncludeMode Mode) {
  // Emit include directive token:
  {
    auto Tag = PTML.enterTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::Directive);
    Tag.finalize();

    PTML.emitLiteral("#include");
  }

  PTML.emitLiteral(" ");

  // Emit include path:
  {
    auto Tag = PTML.enterTag(ptml::tags::Span);
    Tag.emitAttribute(ptml::attributes::Token, ptml::c::tokens::StringLiteral);
    Tag.finalize();

    PTML.emit(Mode == IncludeMode::Quote ? "\"" : "<");
    PTML.emit(Content);
    PTML.emit(Mode == IncludeMode::Quote ? "\"" : ">");
  }

  PTML.emitNewline();
}

void CEmitter::enterScopeImpl(ptml::Emitter::TagEmitter &Tag,
                              Delimiter Delimiter,
                              int Indent,
                              ScopeKind Kind) {
  if (auto Symbols = getDelimiterSymbols(Delimiter))
    emitPunctuator(Symbols->first);

  Tag.initialize(PTML, ptml::tags::Div);
  if (auto Attribute = getScopeKindAttribute(Kind))
    Tag.emitAttribute(ptml::attributes::Scope, *Attribute);
  Tag.finalize();

  PTML.indent(Indent);
}

void CEmitter::leaveScopeImpl(ptml::Emitter::TagEmitter &Tag,
                              Delimiter Delimiter,
                              int Indent) {
  PTML.indent(-Indent);

  Tag.close();

  if (auto Symbols = getDelimiterSymbols(Delimiter))
    emitPunctuator(Symbols->second);
}
