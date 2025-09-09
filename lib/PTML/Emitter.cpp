//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/Emitter.h"

using namespace ptml;

static constexpr llvm::StringRef IndentString = "  ";

template<bool EscapeQuotes = false>
static bool requiresEscaping(char Character) {
  switch (Character) {
  case '<':
  case '>':
  case '&':
    return true;
  case '\"':
    return EscapeQuotes;
  default:
    return false;
  }
}

static llvm::StringRef getEscape(char Character) {
  switch (Character) {
  case '\"':
    return "&quot;";
  case '<':
    return "&lt;";
  case '>':
    return "&gt;";
  case '&':
    return "&amp;";
  default:
    revng_abort("The specified character does not require escaping.");
  }
}

void Emitter::emitLiteral(llvm::StringRef String) {
  revng_assert(CurrentTagEmitter == nullptr);

  constexpr auto Predicate = [](char Character) {
    return Character == '\n' || requiresEscaping(Character);
  };
  revng_assert(std::ranges::none_of(String, Predicate));

  emitLiteralImpl(String);
}

void Emitter::emit(llvm::StringRef String) {
  revng_assert(CurrentTagEmitter == nullptr);

  if (EmitTags)
    emitWithEscapes(String);
  else
    emitWithIndents(String);
}

void Emitter::emitLiteralImpl(llvm::StringRef String) {
  if (not String.empty()) {
    if (HasTrailingNewline)
      emitIndent();

    OS << String;
  }
}

void Emitter::emitWithIndents(llvm::StringRef String) {
  auto View = std::views::split(String, '\n');

  auto Beg = View.begin();
  auto End = View.end();

  if (Beg != End) {
    emitLiteralImpl(std::string_view((*Beg).begin(), (*Beg).end()));

    while (++Beg != End) {
      emitNewline();
      emitLiteralImpl(std::string_view((*Beg).begin(), (*Beg).end()));
    }
  }

  HasTrailingNewline = not String.empty() and String.back() == '\n';
}

template<bool EscapeQuotes>
void Emitter::emitWithEscapes(llvm::StringRef String) {
  auto Beg = String.data();
  auto End = Beg + String.size();

  while (Beg != End) {
    auto Pos = std::find_if(Beg, End, [](char Character) {
      return requiresEscaping<EscapeQuotes>(Character);
    });

    emitWithIndents(std::string_view(Beg, Pos));

    if (Pos != End)
      OS << getEscape(*Pos++);

    Beg = Pos;
  }
}

void Emitter::emitIndentWhitespace() {
  for (unsigned I = 0, C = Indent; I < C; ++I)
    OS << IndentString;
}

void Emitter::emitIndentWithTags() {
  auto Tag = enterTag(ptml::tags::Span);
  Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Indentation);
  Tag.finalize();

  emitIndentWhitespace();
}

void Emitter::emitIndent() {
  HasTrailingNewline = false;

  if (Indent != 0) {
    if (EmitTags)
      emitIndentWithTags();
    else
      emitIndentWhitespace();
  }
}

void Emitter::emitAttributeValue(llvm::StringRef String) {
  emitWithEscapes</*EscapeQuotes=*/true>(String);
}
