//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"
#include "revng/Support/Assert.h"

using namespace ptml;

namespace {

// PTML requires escaping some characters. Currently we escape angle brackets
// and ampersands unconditionally. Quotes are escaped only within attribute
// values, which are themselves delimited by quotes. Attribute values delimited
// by apostrophes are not emitted, so there is no need to ever escape them.
//
// In some situations escaping angle brackets could be avoided, but these
// situations are either not encountered in practice or introduce asymmetries.
// For this reason they are escaped unconditionally.

static bool requiresEscaping(char Character, bool EscapeQuotes) {
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

static void
emitEscaped(StreamEmitter &Emitter, llvm::StringRef String, bool EscapeQuotes) {
  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [EscapeQuotes](char Character) {
      return requiresEscaping(Character, EscapeQuotes);
    });

    Emitter.emit(llvm::StringRef(std::string_view(Begin, Pos)));

    if (Pos != End)
      Emitter.emit(getEscape(*Pos++));

    Begin = Pos;
  }
}

} // namespace

//===--------------------------- PTMLTagEmitter ---------------------------===//

PTMLTagEmitter::PTMLTagEmitter(PTMLStreamEmitter &Parent, llvm::StringRef Tag) :
  ParentEmitter(Parent), Tag(Tag) {
  revng_assert(ParentEmitter.CurrentOpenTagEmitter == nullptr,
               "The parent emitter is already associated with an unfinalized "
               "open tag.");

  if (ParentEmitter.EmitTags)
    ParentEmitter.OS << '<' << Tag;
  ParentEmitter.CurrentOpenTagEmitter = this;
}

PTMLTagEmitter::~PTMLTagEmitter() {
  if (IsEmittingOpenTag)
    finalizeOpenTag();

  if (ParentEmitter.EmitTags)
    ParentEmitter.OS << '<' << '/' << Tag << '>';
}

void PTMLTagEmitter::finalizeOpenTag() {
  revng_assert(IsEmittingOpenTag, "The open tag has already been finalized.");

  if (ParentEmitter.EmitTags)
    ParentEmitter.OS << '>';

  IsEmittingOpenTag = false;
  ParentEmitter.CurrentOpenTagEmitter = nullptr;
}

void PTMLTagEmitter::emitAttributeValue(llvm::StringRef Value) {
  emitEscaped(ParentEmitter, Value, /*EscapeQuotes=*/true);
}

PTMLTagEmitter &PTMLTagEmitter::emitAttribute(llvm::StringRef Name,
                                              llvm::StringRef Value) {
  revng_assert(ParentEmitter.CurrentOpenTagEmitter == this);
  revng_assert(IsEmittingOpenTag, "The open tag has already been finalized.");

  revng_assert(not Name.contains('\n'));
  revng_assert(not Value.contains('\n'));

  if (ParentEmitter.EmitTags) {
    ParentEmitter.OS << ' ' << Name << '=' << '"';
    emitAttributeValue(Value);
    ParentEmitter.OS << '"';
  }

  return *this;
}

PTMLTagEmitter &
PTMLTagEmitter::emitListAttribute(llvm::StringRef Name,
                                  llvm::ArrayRef<llvm::StringRef> Values) {
  revng_assert(ParentEmitter.CurrentOpenTagEmitter == this);
  revng_assert(IsEmittingOpenTag, "The open tag has already been finalized.");

  revng_assert(not Name.contains('\n'));
  revng_assert(std::ranges::none_of(Values, [](llvm::StringRef String) {
    return String.contains('\n');
  }));

  if (ParentEmitter.EmitTags) {
    ParentEmitter.OS << ' ' << Name << '=' << '"';

    for (auto [I, Value] : llvm::enumerate(Values)) {
      revng_assert(not Value.contains(','),
                   "List attribute values shall not contain commas.");

      if (I != 0)
        ParentEmitter.OS << ',';

      emitAttributeValue(Value);
    }

    ParentEmitter.OS << '"';
  }

  return *this;
}

//===-------------------------- PTMLStreamEmitter -------------------------===//

void PTMLStreamEmitter::emit(llvm::StringRef Content) {
  revng_assert(CurrentOpenTagEmitter == nullptr,
               "Cannot emit content while an unfinalized tag emitter is "
               "associated with this emitter.");

  if (EmitTags)
    emitEscaped(*this, Content, /*EscapeQuotes=*/false);
  else
    StreamEmitter::emit(Content);
}
