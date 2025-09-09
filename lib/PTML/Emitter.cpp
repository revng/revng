//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/Emitter.h"

using namespace ptml;

static constexpr llvm::StringRef IndentString = "  ";

// PTML requires escaping some characters. Currently we escape angle brackets
// and ampersands unconditionally. Quotes are escaped only within attribute
// values, which are themselves delimited by quotes. Attribute values delimited
// by apostrophes are not emitted, so there is no need to ever escape them.
//
// In some situations escaping angle brackets could be avoided, but these
// situations are either not encountered in practice or introduce asymmetries.
// For this reason they are escaped unconditionally.

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

//===------------------------------- Emitter ------------------------------===//

void Emitter::emitLiteralContent(llvm::StringRef String) {
  revng_assert(CurrentOpenTagEmitter == nullptr,
               "Cannot emit content while an unfinalized TagEmitter is "
               "associated with this Emitter.");

  constexpr auto IsNewlineOrRequiresEscaping = [](char Character) {
    return Character == '\n' || requiresEscaping(Character);
  };
  revng_assert(std::ranges::none_of(String, IsNewlineOrRequiresEscaping));

  emitLiteralContentImpl(String);
}

void Emitter::emitContent(llvm::StringRef String) {
  revng_assert(CurrentOpenTagEmitter == nullptr,
               "Cannot emit content while an unfinalized TagEmitter is "
               "associated with this Emitter.");

  if (EmitTags)
    emitEscapedContent(String);
  else
    emitIndentedContent(String);
}

void Emitter::emitLiteralContentImpl(llvm::StringRef String) {
  if (not String.empty()) {
    if (IsAtBeginningOfLine)
      emitIndentation();

    OS << String;
  }
}

void Emitter::emitIndentedContent(llvm::StringRef String) {
  if (not String.empty()) {
    auto View = std::views::split(String, '\n');

    auto Begin = View.begin();
    auto End = View.end();
    revng_assert(Begin != End);

    emitLiteralContentImpl(std::string_view((*Begin).begin(), (*Begin).end()));

    while (++Begin != End) {
      emitContentNewline();
      emitLiteralContentImpl(std::string_view((*Begin).begin(),
                                              (*Begin).end()));
    }

    IsAtBeginningOfLine = String.back() == '\n';
  }
}

template<bool EscapeQuotes>
void Emitter::emitEscapedContent(llvm::StringRef String) {
  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [](char Character) {
      return requiresEscaping<EscapeQuotes>(Character);
    });

    emitIndentedContent(std::string_view(Begin, Pos));

    if (Pos != End)
      OS << getEscape(*Pos++);

    Begin = Pos;
  }
}

void Emitter::emitIndentation() {
  IsAtBeginningOfLine = false;

  if (Indentation != 0) {
    TagEmitter Tag;

    if (EmitTags) {
      Tag.initializeOpenTag(*this, ptml::tags::Span);
      Tag.emitAttribute(ptml::attributes::Token, ptml::tokens::Indentation);
      Tag.finalizeOpenTag();
    }

    for (unsigned I = 0, C = Indentation; I < C; ++I)
      OS << IndentString;
  }
}

void Emitter::emitAttributeValue(llvm::StringRef String) {
  emitEscapedContent</*EscapeQuotes=*/true>(String);
}

//===------------------------- Emitter::TagEmitter ------------------------===//

void TagEmitter::initializeOpenTagImpl(Emitter &ParentEmitter,
                                       llvm::StringRef Tag) {
  this->ParentEmitter = &ParentEmitter;
  this->Tag = Tag;
  this->IsOpenTagFinalized = false;

  if (ParentEmitter.EmitTags) {
    if (ParentEmitter.IsAtBeginningOfLine)
      ParentEmitter.emitIndentation();
    ParentEmitter.OS << '<' << Tag;
  }

  ParentEmitter.CurrentOpenTagEmitter = this;
}

void TagEmitter::emitAttributeImpl(llvm::StringRef Name,
                                   llvm::StringRef Value) {
  revng_assert(ParentEmitter->CurrentOpenTagEmitter == this);

  if (ParentEmitter->EmitTags) {
    ParentEmitter->OS << ' ' << Name << '=' << '"';
    ParentEmitter->emitAttributeValue(Value);
    ParentEmitter->OS << '"';
  }
}

void TagEmitter::emitListAttributeImpl(llvm::StringRef Name,
                                       llvm::ArrayRef<llvm::StringRef> Values) {
  revng_assert(ParentEmitter->CurrentOpenTagEmitter == this);

  if (ParentEmitter->EmitTags) {
    ParentEmitter->OS << ' ' << Name << '=' << '"';

    bool InsertComma = false;
    for (llvm::StringRef Value : Values) {
      revng_assert(not Value.contains(','),
                   "List attribute values shall not contain commas.");

      if (InsertComma)
        ParentEmitter->OS << ',';
      InsertComma = true;

      ParentEmitter->emitAttributeValue(Value);
    }

    ParentEmitter->OS << '"';
  }
}

void TagEmitter::finalizeOpenTagImpl() {
  if (not IsOpenTagFinalized) {
    revng_assert(ParentEmitter->CurrentOpenTagEmitter == this);

    if (ParentEmitter->EmitTags)
      ParentEmitter->OS << '>';
    IsOpenTagFinalized = true;

    ParentEmitter->CurrentOpenTagEmitter = nullptr;
  }
}

void TagEmitter::closeImpl() {
  if (ParentEmitter != nullptr) {
    finalizeOpenTagImpl();

    if (ParentEmitter->EmitTags)
      ParentEmitter->OS << '<' << '/' << Tag << '>';
  }
  ParentEmitter = nullptr;
}
