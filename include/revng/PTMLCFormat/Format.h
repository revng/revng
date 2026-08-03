#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstddef>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace ptml {

/// A single element tag of a PTML document, identified by its byte range in it.
/// IsClosing distinguishes end tags (`</...>`) from start tags.
struct PTMLTagRange {
  size_t Begin = 0;
  size_t End = 0;
  bool IsClosing = false;
};

/// Represents a run of source characters that map one-to-one onto consecutive
/// PTML bytes: the character at SourceOffset sits at PTMLOffset, and every
/// following character advances both offsets by one, until the next sync
/// point.
///
/// Sync points are recorded only where this correspondence breaks (a
/// tag, or an escaped character), so there is no entry per character.
struct OffsetSyncPoint {
  size_t SourceOffset = 0;
  size_t PTMLOffset = 0;
};

/// Describes a PTML document in terms of the C source underlying it and how it
/// maps back onto the markup. This is everything reformat needs to map the
/// edits produced on the recovered source back onto the tagged document.
///
/// It is built by the emitter as it writes the document, so reformatting never
/// has to parse the PTML back.
struct PTMLCReformattableDocument {
  /// The C source underlying the document: its text content, unescaped, with
  /// the markup removed.
  std::string Source;

  /// The source-to-PTML sync points, in ascending SourceOffset order (see
  /// OffsetSyncPoint).
  std::vector<OffsetSyncPoint> SourceMap;

  /// The document's tags, in document order.
  std::vector<PTMLTagRange> Tags;

  /// Reformats a PTML document with clang-format, preserving all of its tags.
  /// PTML must be the document this metadata was built from: the recovered
  /// source is formatted and the resulting whitespace-only edits are mapped
  /// back onto the tags. E.g. "<span>foo</span> <span>(</span>", whose
  /// recovered source "foo (" clang-format rewrites to "foo(", becomes
  /// "<span>foo</span><span>(</span>".
  std::string reformat(llvm::StringRef PTML) const;

private:
  /// The PTML byte offset the given source offset maps to.
  size_t ptmlOffset(size_t SourceOffset) const;
};

/// Reformats plain C source using clang-format.
std::string reformatC(llvm::StringRef Source);

} // namespace ptml
