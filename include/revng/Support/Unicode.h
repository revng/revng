#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Unicode.h"

#include "revng/Support/Assert.h"

class UnicodeCStringView {
public:
  /// The enumerators are ordered by preference: when a buffer decodes as more
  /// than one of them, the first is the reading to favor.
  enum class Encoding {
    Invalid,
    UTF8,
    UTF16LE,
    UTF16BE
  };

  using CodePointProcessor = bool (*)(size_t Offset,
                                      size_t CodePointIndex,
                                      uint32_t CodePoint);

  /// The readings of a buffer, at most one per encoding, most preferred first.
  using CandidateList = llvm::SmallVector<UnicodeCStringView, 3>;

private:
  llvm::StringRef Data;
  Encoding TheEncoding = Encoding::Invalid;
  size_t CodePointCount = 0;

public:
  UnicodeCStringView() :
    UnicodeCStringView({}, UnicodeCStringView::Encoding::Invalid, 0) {}

  UnicodeCStringView(llvm::StringRef Data,
                     Encoding TheEncoding,
                     size_t CodePointCount);

public:
  /// Every string \p Data could spell, whatever its length.
  ///
  /// A buffer often spells more than one: `"a\0\0"` is a one character UTF-8
  /// string and a one character UTF-16 one at the same time. Telling them
  /// apart takes context this function does not have, so it reports all the
  /// readings and leaves the choice to the caller.
  static CandidateList getPrintable(llvm::ArrayRef<uint8_t> Data);

  static CandidateList get(llvm::ArrayRef<uint8_t> Data,
                           CodePointProcessor ProcessCodePoint);

  static UnicodeCStringView fromUTF8(llvm::ArrayRef<uint8_t> Data,
                                     CodePointProcessor ProcessCodePoint);

  template<bool IsLittleEndian>
  static UnicodeCStringView
  fromUTF16(llvm::ArrayRef<uint8_t> Data, CodePointProcessor ProcessCodePoint);

public:
  bool isValid() const { return TheEncoding != Encoding::Invalid; }
  llvm::StringRef data() const { return Data; }
  Encoding encoding() const { return TheEncoding; }
  size_t codePointCount() const { return CodePointCount; }

  /// \return 1 for UTF8, 2 for UTF16.
  unsigned charSize() const {
    revng_assert(TheEncoding != Encoding::Invalid);
    return TheEncoding == Encoding::UTF8 ? 1 : 2;
  }
};

extern template UnicodeCStringView
UnicodeCStringView::fromUTF16<true>(llvm::ArrayRef<uint8_t> Data,
                                    CodePointProcessor ProcessCodePoint);

extern template UnicodeCStringView
UnicodeCStringView::fromUTF16<false>(llvm::ArrayRef<uint8_t> Data,
                                     CodePointProcessor ProcessCodePoint);
