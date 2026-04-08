//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PTML/Constants.h"
#include "revng/PTML/PTMLEmitter.h"

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

template<bool EscapeQuotes>
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

template<Emitter EmitterT, bool EscapeQuotes>
static void
emitEscaped(std::type_identity_t<EmitterT> &Emitter, llvm::StringRef String) {
  auto Begin = String.data();
  auto End = Begin + String.size();

  while (Begin != End) {
    auto Pos = std::find_if(Begin, End, [](char Character) {
      return requiresEscaping<EscapeQuotes>(Character);
    });

    Emitter.emit(llvm::StringRef(std::string_view(Begin, Pos)));

    if (Pos != End)
      Emitter.emit(getEscape(*Pos++));

    Begin = Pos;
  }
}

} // namespace

//===--------------------------- PTMLTagEmitter ---------------------------===//

PTMLTagEmitter::PTMLTagEmitter(detail::PTMLEmitterBase &ParentEmitter,
                               llvm::StringRef Tag) :
  ParentEmitter(ParentEmitter), Tag(Tag) {
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
  emitEscaped<StreamEmitter, /*EscapeQuotes=*/true>(ParentEmitter, Value);
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

    bool InsertComma = false;
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
    emitEscaped<IndentingEmitter, /*EscapeQuotes=*/false>(*this, Content);
  else
    IndentingEmitter::emit(Content);
}
