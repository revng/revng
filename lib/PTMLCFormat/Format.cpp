//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "clang/Format/Format.h"
#include "clang/Tooling/Core/Replacement.h"

#include "revng/PTML/CAttributes.h"
#include "revng/PTMLCFormat/Format.h"
#include "revng/Support/Assert.h"

using ptml::OffsetSyncPoint;
using ptml::PTMLTagRange;

namespace {

/// Produces the whitespace-only edits clang-format would apply to Code.
static clang::tooling::Replacements
getFormattingReplacements(llvm::StringRef Code) {
  clang::format::FormatStyle Style = clang::format::getLLVMStyle();

  // A column limit of zero disables line wrapping, so that only indentation
  // and spacing are normalized, without reflowing existing lines.
  Style.ColumnLimit = 0;

  // Reordering includes is not a whitespace-only edit: it would rewrite content
  // and corrupt the mapping back onto the tags. Keep includes as emitted.
  Style.SortIncludes = clang::format::FormatStyle::SI_Never;

  // The LLVM style drops the space after a C-style cast, but the emitter writes
  // it; keep it rather than replaying its removal as a whitespace edit.
  Style.SpaceAfterCStyleCast = true;

  // Teach clang-format the attribute-like macros the emitter writes, from their
  // single registry, so it parses the declarations they decorate rather than
  // misreading the run of macros. In particular this keeps the space before the
  // brace of an enum written as
  // "enum _ENUM_UNDERLYING(t) _PACKED name { ... }".
  for (const ptml::Attribute &Entry : ptml::Attributes.StaticAttributes)
    Style.AttributeMacros.push_back(std::string(Entry.Macro));
  for (const ptml::Annotation &Entry : ptml::Attributes.StaticAnnotations)
    Style.AttributeMacros.push_back(std::string(Entry.Macro));

  std::vector<clang::tooling::Range> Ranges;
  Ranges.emplace_back(0, Code.size());

  return clang::format::reformat(Style, Code, Ranges, "input.c");
}

/// Extends Begin leftward over any opening tags flush against the range (a tag
/// whose End equals Begin), so the replacement whitespace is emitted *before*
/// them and stays outside the elements they open.
///
/// Example where this fires: clang-format inserts a newline between "a" and "b"
/// of "ab", emitted as "<span>a</span><span>b</span>". The insertion maps to
/// the "b", i.e. right after the second <span>'s end. Left as is, the newline
/// would land inside b's span ("<span>a</span><span>\nb</span>"); pulling Begin
/// left across that <span> makes spliceTags emit it before the tag instead
/// ("<span>a</span>\n<span>b</span>").
static size_t
expandLeftPastOpeningTags(size_t Begin,
                          const llvm::DenseMap<size_t, const PTMLTagRange *>
                            &EndToTag) {
  while (true) {
    auto Iterator = EndToTag.find(Begin);
    if (Iterator == EndToTag.end() or Iterator->second->IsClosing)
      break;
    Begin = Iterator->second->Begin;
  }
  return Begin;
}

/// Extends End rightward over any closing tags flush against the range (a tag
/// whose Begin equals End), so the replacement whitespace is emitted *after*
/// them and stays outside the elements they close. This is the mirror of
/// expandLeftPastOpeningTags: e.g. a newline inserted just before a "</div>"
/// that ends a scope is pushed out to after that tag.
static size_t
expandRightPastClosingTags(size_t End,
                           const llvm::DenseMap<size_t, const PTMLTagRange *>
                             &StartToTag) {
  while (true) {
    auto Iterator = StartToTag.find(End);
    if (Iterator == StartToTag.end() or not Iterator->second->IsClosing)
      break;
    End = Iterator->second->End;
  }
  return End;
}

/// Rebuilds the PTML byte range [Begin, End) with its whitespace replaced by
/// Replacement, keeping every tag in the range. Leading closing tags stay
/// before the new whitespace and everything else after it, so the whitespace
/// ends up between elements rather than inside one. StartToTag maps a tag's
/// start offset to the tag.
///
/// The range holds only whitespace and whole tags (no partial tags or other
/// content) and never reorders them, so any close/open run works. For example
/// "\n</div><div></div>\n" yields "</div>" + Replacement + "<div></div>": the
/// empty element keeps its own </div> in After, beside its <div>.
///
/// Running example, continuing PTMLCReformattableDocument::reformat: the range
/// is PTML [8, 16) = "\n</div>\n" and Replacement is "\n". Walking Cursor from
/// 8 to 16:
///
///   Cursor=8 : nothing starts at 8, a whitespace byte -> dropped;  -> 9
///   Cursor=9 : </div> [9,15) starts here; closing and still
///              leading, so it goes to Before;                      -> 15
///   Cursor=15: nothing starts at 15, a whitespace byte -> dropped; -> 16
///   Cursor=16: Cursor == End -> stop.
///
/// Before is "</div>", After is "", so it returns "</div>" + "\n" + "" =
/// "</div>\n": the closing tag is preserved and the blank line collapses.
static std::string
spliceTags(llvm::StringRef PTML,
           const llvm::DenseMap<size_t, const PTMLTagRange *> &StartToTag,
           size_t Begin,
           size_t End,
           llvm::StringRef Replacement) {
  std::string Before;
  std::string After;
  bool InLeadingClosingRun = true;

  for (size_t Cursor = Begin; Cursor < End;) {
    auto Iterator = StartToTag.find(Cursor);
    if (Iterator == StartToTag.end()) {
      // Not the start of a tag, so a whitespace byte being replaced: drop it.
      ++Cursor;
      continue;
    }

    // A tag: keep it. Leading closing tags belong before the new whitespace;
    // the first opening tag, and everything after it, belongs after.
    const PTMLTagRange &Tag = *Iterator->second;
    llvm::StringRef Text = PTML.slice(Tag.Begin, Tag.End);
    if (InLeadingClosingRun and Tag.IsClosing) {
      Before.append(Text.data(), Text.size());
    } else {
      InLeadingClosingRun = false;
      After.append(Text.data(), Text.size());
    }

    Cursor = Tag.End;
  }

  return Before + Replacement.str() + After;
}

} // namespace

namespace ptml {

size_t PTMLCReformattableDocument::ptmlOffset(size_t SourceOffset) const {
  // SourceMap is sorted by SourceOffset and its first sync point is always at
  // offset 0, so there is a sync point at or before any queried offset. Find
  // the last such sync point and extend its one-to-one run up to the query.
  auto Iterator = llvm::upper_bound(SourceMap,
                                    SourceOffset,
                                    [](size_t Value,
                                       const OffsetSyncPoint &SyncPoint) {
                                      return Value < SyncPoint.SourceOffset;
                                    });
  --Iterator;
  return Iterator->PTMLOffset + (SourceOffset - Iterator->SourceOffset);
}

std::string PTMLCReformattableDocument::reformat(llvm::StringRef PTML) const {
  // clang-format edits the recovered Source (the plain C, with tags stripped
  // and escapes undone). Each edit replaces a run of whitespace; this function
  // replays those edits on the tagged PTML instead, so the tags are kept.
  //
  // Running example. Say the emitter produced this 30-byte PTML:
  //              111111 111122222222223
  //   012345678 9012345 678901234567890
  //   <div>foo\n</div>\n<div>bar</div>
  //
  // with tags <div> [0,5), </div> [9,15), <div> [16,21), </div> [24,30). The
  // two "\n" bytes are then at offsets 8 and 15, and the recovered Source (the
  // same text with the four tags removed) is "foo\n\nbar". clang-format asks to
  // collapse the blank line: replace Source "\n\n" (offset 3, length 2) with
  // "\n". We follow that single edit through the code.
  clang::tooling::Replacements Replacements = getFormattingReplacements(Source);
  // -> Replacements = { at Source offset 3, replace 2 bytes with "\n" }

  // Index every tag by the byte offset it starts and ends at.
  llvm::DenseMap<size_t, const PTMLTagRange *> StartToTag;
  llvm::DenseMap<size_t, const PTMLTagRange *> EndToTag;
  for (const PTMLTagRange &Tag : Tags) {
    StartToTag.try_emplace(Tag.Begin, &Tag);
    EndToTag.try_emplace(Tag.End, &Tag);
  }
  // -> StartToTag = { 0:<div>, 9:</div>, 16:<div>, 24:</div> }
  // -> EndToTag   = { 5:<div>, 15:</div>, 21:<div>, 30:</div> }

  // Apply the edits in one forward pass, copying the untouched PTML in between.
  // Replacements come in ascending Source offset, which maps to ascending PTML
  // offsets, so Position only moves forward.
  std::string Result;
  Result.reserve(PTML.size());
  size_t Position = 0;

  for (const clang::tooling::Replacement &Replacement : Replacements) {
    size_t Offset = Replacement.getOffset(); // -> 3
    size_t Length = Replacement.getLength(); // -> 2

    // Map the replaced Source range onto a PTML byte range. Whitespace is never
    // escaped nor placed inside a tag, so each replaced byte maps one-to-one:
    // Begin is the PTML offset of the first replaced byte, End one past the
    // last. In between lie only the replaced whitespace bytes and whole tags.
    size_t Begin = ptmlOffset(Offset); // -> 8
    size_t End = Length == 0 ? Begin : ptmlOffset(Offset + Length - 1) + 1;
    // -> End = ptmlOffset(4) + 1 = 16, so [Begin, End) = PTML [8, 16) =
    //    "\n</div>\n": here the whitespace run straddles the first </div>.

    // Nudge the range past any tags flush against it, so the new whitespace
    // lands between elements. Both are no-ops here: no tag ends at 8, and the
    // tag starting at 16 is opening, not closing.
    Begin = expandLeftPastOpeningTags(Begin, EndToTag); // -> 8
    End = expandRightPastClosingTags(End, StartToTag); // -> 16

    // Whitespace-only edits never overlap: mapped to PTML they stay ordered,
    // and the nudges above only ever meet flush ranges, never cross them, so
    // this edit always begins at or after the previous one ended.
    revng_assert(Begin >= Position);

    // Replaying an edit on the PTML only drops the whitespace in its range, so
    // anything but whitespace coming back would be silently lost.
    llvm::StringRef ReplacementText = Replacement.getReplacementText();
    revng_assert(llvm::all_of(ReplacementText, llvm::isSpace),
                 "clang-format produced a non-whitespace replacement.");

    // Rebuild the range: drop its whitespace, keep its tags. Here this returns
    // "</div>\n" (the straddled closing tag stays left of the new "\n").
    std::string Rebuilt = spliceTags(PTML,
                                     StartToTag,
                                     Begin,
                                     End,
                                     ReplacementText);

    // Copy the untouched PTML up to the edit, then the rebuilt range.
    Result.append(PTML.data() + Position, Begin - Position); // -> "<div>foo"
    Result.append(Rebuilt); // -> + "</div>\n"
    Position = End; // -> 16
  }

  // Copy whatever PTML follows the last edit.   -> "<div>bar</div>"
  Result.append(PTML.data() + Position, PTML.size() - Position);
  // -> Result = "<div>foo</div>\n<div>bar</div>" (its Source is now "foo\nbar")

  return Result;
}

std::string reformatC(llvm::StringRef Source) {
  clang::tooling::Replacements Replacements = getFormattingReplacements(Source);

  // The replacements come from clang-format run on this exact source, so
  // applying them back can only fail on a bug, not on the input; surface that
  // as an abort rather than silently returning unformatted text.
  return llvm::cantFail(clang::tooling::applyAllReplacements(Source,
                                                             Replacements));
}

} // namespace ptml
