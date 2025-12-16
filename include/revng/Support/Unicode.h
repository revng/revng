#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Support/Unicode.h"

#include "revng/Support/Assert.h"

class UnicodeCStringView {
public:
  enum class Encoding {
    Invalid,
    UTF8,
    UTF16LE,
    UTF16BE
  };

  using CodePointProcessor = bool (*)(size_t Offset,
                                      size_t CodePointIndex,
                                      uint32_t CodePoint);

private:
  llvm::StringRef Data;
  Encoding Encoding = Encoding::Invalid;
  size_t CodePointCount = 0;

public:
  UnicodeCStringView() :
    UnicodeCStringView({}, UnicodeCStringView::Encoding::Invalid, 0) {}

  UnicodeCStringView(llvm::StringRef Data,
                     enum Encoding Encoding,
                     size_t CodePointCount);

public:
  static UnicodeCStringView getPrintable(llvm::ArrayRef<uint8_t> Data);

  static UnicodeCStringView get(llvm::ArrayRef<uint8_t> Data,
                                CodePointProcessor ProcessCodePoint);

  static UnicodeCStringView fromUTF8(llvm::ArrayRef<uint8_t> Data,
                                     CodePointProcessor ProcessCodePoint);

  template<bool IsLittleEndian>
  static UnicodeCStringView
  fromUTF16(llvm::ArrayRef<uint8_t> Data, CodePointProcessor ProcessCodePoint);

public:
  bool isValid() const { return Encoding != Encoding::Invalid; }
  llvm::StringRef data() const { return Data; }
  enum Encoding encoding() const { return Encoding; }
  size_t codePointCount() const { return CodePointCount; }

  /// \return 1 for UTF8, 2 for UTF16.
  unsigned charSize() const {
    revng_assert(Encoding != Encoding::Invalid);
    return Encoding == Encoding::UTF8 ? 1 : 2;
  }
};

extern template UnicodeCStringView
UnicodeCStringView::fromUTF16<true>(llvm::ArrayRef<uint8_t> Data,
                                    CodePointProcessor ProcessCodePoint);

extern template UnicodeCStringView
UnicodeCStringView::fromUTF16<false>(llvm::ArrayRef<uint8_t> Data,
                                     CodePointProcessor ProcessCodePoint);
