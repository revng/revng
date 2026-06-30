//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"

#include "revng/Support/Debug.h"
#include "revng/Support/Unicode.h"

using namespace llvm;

static Logger Log("unicode");

using CodePointProcessor = UnicodeCStringView::CodePointProcessor;

UnicodeCStringView::UnicodeCStringView(llvm::StringRef Data,
                                       enum Encoding Encoding,
                                       size_t CodePointCount) :
  Data(Data), Encoding(Encoding), CodePointCount(CodePointCount) {

  // Validate
  switch (Encoding) {
  case Encoding::Invalid:
    revng_assert(Data.data() == nullptr);
    revng_assert(Data.empty());
    revng_assert(CodePointCount == 0);
    break;
  case Encoding::UTF8: {
    revng_assert(Data.data() != nullptr);
    revng_assert(CodePointCount > 0);
    revng_assert(Data.size() >= CodePointCount);
    StringRef NullString("\x00", 1);
    revng_assert(Data.ends_with(NullString));
    revng_assert(not Data.drop_back().contains(NullString));
  } break;
  case Encoding::UTF16LE:
  case Encoding::UTF16BE: {
    revng_assert(Data.size() % 2 == 0);
    revng_assert(Data.data() != nullptr);
    revng_assert(CodePointCount > 0);
    revng_assert(Data.size() >= 2 * CodePointCount);
    StringRef NullString("\x00\x00", 2);
    revng_assert(Data.ends_with(NullString));

    // Check if we have a 2-bytes aligned NullString
    StringRef StringWithoutNUL = Data.drop_back(2);
    size_t Index = StringWithoutNUL.find(NullString, 0);
    // find() == 1
    // AA 00 00: size 3, do not proceed
    // AA 00 00 ??: size 4, proceed
    while (Index != StringRef::npos or StringWithoutNUL.size() < Index + 3) {
      revng_assert(Index % 2 == 1);
      Index = StringWithoutNUL.find(NullString, Index + 1);
    }
  } break;
  }
}

template<bool IsLittleEndian>
UnicodeCStringView
UnicodeCStringView::fromUTF16(ArrayRef<uint8_t> Data,
                              CodePointProcessor ProcessCodePoint) {
  revng_log(Log, "Decoding as UTF16");
  LoggerIndent Indent(Log);

  const unsigned char *Start = Data.data();
  const unsigned char *Current = Data.data();
  const unsigned char *End = Data.end();
  size_t CodePointIndex = 0;

  while (Current + 1 < End) {
    revng_log(Log, "Processing code point #" << CodePointIndex);
    LoggerIndent Indent(Log);

    // Extract 16-bit code unit according to endianness
    uint16_t Unit = 0;
    uint16_t FirstByte = Current[0];
    uint16_t SecondByte = Current[1];
    if constexpr (IsLittleEndian)
      Unit = FirstByte | (SecondByte << 8);
    else
      Unit = SecondByte | (FirstByte << 8);

    // Proceed
    Current += 2;

    if (Unit == 0) {
      // We've found the terminator
      if (Current - 2 == Start) {
        revng_log(Log, "Empty string, bailing out");
        return {};
      } else {
        unsigned long Size = Current - Start;
        revng_log(Log, "Detected a string of " << Size << " bytes");
        return UnicodeCStringView({ reinterpret_cast<const char *>(Data.data()),
                                    Size },
                                  IsLittleEndian ? Encoding::UTF16LE :
                                                   Encoding::UTF16BE,
                                  CodePointIndex);
      }
    }

    uint32_t CodePoint = 0;

    if (Unit < 0xD800 or Unit > 0xDFFF) {
      // BMP code point
      CodePoint = Unit;
    } else if ((Unit & 0b1111110000000000) == 0b1101100000000000) {
      // High surrogate
      if (Current + 1 >= End) {
        revng_log(Log, "Truncated surrogate pair, bailing out");
        return {};
      }

      uint16_t LowUnit = 0;
      if constexpr (IsLittleEndian)
        LowUnit = FirstByte | (SecondByte << 8);
      else
        LowUnit = SecondByte | (FirstByte << 8);

      Current += 2;

      if ((LowUnit & 0b1111110000000000) != 0b1101110000000000) {
        revng_log(Log, "Invalid low surrogate, bailing out");
        return {};
      }

      const uint16_t Low10Bits = 0b0000001111111111;
      CodePoint = 0x10000
                  + ((static_cast<uint32_t>(Unit & Low10Bits) << 10)
                     | (LowUnit & Low10Bits));
    } else {
      revng_log(Log, "Lone low surrogate, bailing out");
      return {};
    }

    // Reject codepoints above Unicode range
    if (CodePoint > 0x10FFFF) {
      revng_log(Log, "Codepoint out of range, bailing out");
      return {};
    }

    if (not ProcessCodePoint(Current - Start, CodePointIndex, CodePoint)) {
      revng_log(Log, "Codepoint rejected, bailing out");
      return {};
    }

    ++CodePointIndex;
  }

  // Reached the end without null terminator
  revng_log(Log, "No NUL terminator found, bailing out");
  return {};
}

template UnicodeCStringView
UnicodeCStringView::fromUTF16<true>(llvm::ArrayRef<uint8_t> Data,
                                    CodePointProcessor ProcessCodePoint);
template UnicodeCStringView
UnicodeCStringView::fromUTF16<false>(llvm::ArrayRef<uint8_t> Data,
                                     CodePointProcessor ProcessCodePoint);

UnicodeCStringView
UnicodeCStringView::fromUTF8(ArrayRef<uint8_t> Data,
                             CodePointProcessor ProcessCodePoint) {
  const unsigned char *Start = Data.data();
  const unsigned char *Current = Data.data();
  const unsigned char *End = Data.end();
  size_t CodePointIndex = 0;

  while (Current < End) {
    unsigned char Byte = *Current;
    uint32_t CodePoint = 0;
    size_t ExtraBytes = 0;

    if (Byte == 0) {
      // We found a NUL byte!
      if (Current == Start) {
        return {};
      } else {
        unsigned long Size = Current - Start + 1;
        StringRef Result(reinterpret_cast<const char *>(Data.data()), Size);
        return UnicodeCStringView(Result, Encoding::UTF8, CodePointIndex);
      }
    } else if ((Byte & 0b10000000) == 0) {
      CodePoint = Byte;
      ExtraBytes = 0;
    } else if ((Byte & 0b11100000) == 0b11000000) {
      CodePoint = Byte & 0b00011111;
      ExtraBytes = 1;
    } else if ((Byte & 0b11110000) == 0b11100000) {
      CodePoint = Byte & 0b00001111;
      ExtraBytes = 2;
    } else if ((Byte & 0b11111000) == 0b11110000) {
      CodePoint = Byte & 0b00000111;
      ExtraBytes = 3;
    } else {
      // Invalid start byte
      return {};
    }

    if (Current + ExtraBytes > End) {
      // Truncated sequence
      return {};
    }

    // Consume 6 (less significant) bits for each extra byte
    for (size_t I = 0; I < ExtraBytes; ++I) {
      Byte = *(Current + 1 + I);
      if ((Byte & 0b11000000) != 0b10000000) {
        // The top bits of extra bytes are not 0b10
        return {};
      }
      CodePoint = (CodePoint << 6) | (Byte & 0b00111111);
    }

    // Reject overlong encodings and invalid ranges
    if (CodePoint < 0x80 && ExtraBytes > 0)
      return {};
    else if (CodePoint < 0x800 && ExtraBytes > 1)
      return {};
    else if (CodePoint < 0x10000 && ExtraBytes > 2)
      return {};
    else if (CodePoint >= 0xD800 && CodePoint <= 0xDFFF)
      return {};
    else if (CodePoint > 0x10FFFF)
      return {};

    if (not ProcessCodePoint(Current - Start, CodePointIndex, CodePoint))
      return {};

    Current += 1 + ExtraBytes;
    ++CodePointIndex;
  }

  // Reached the end without null terminator
  return {};
}

UnicodeCStringView
UnicodeCStringView::get(ArrayRef<uint8_t> Data,
                        CodePointProcessor ProcessCodePoint) {
  using Decoder = UnicodeCStringView (*)(ArrayRef<uint8_t>, CodePointProcessor);
  static const std::array<std::pair<StringRef, Decoder>, 3> Decoders = {
    { { "UTF8", UnicodeCStringView::fromUTF8 },
      { "UTF16LE", UnicodeCStringView::fromUTF16<true> },
      { "UTF16BE", UnicodeCStringView::fromUTF16<false> } }
  };

  if (Log.isEnabled()) {
    revng_log(Log, "Trying to decode a buffer of size " << Data.size() << ": ");
    const char *Pointer = reinterpret_cast<const char *>(Data.data());
    llvm::printEscapedString({ Pointer, Data.size() }, *Log.getAsLLVMStream());
    Log << DoLog;
  }
  LoggerIndent Indent(Log);

  for (auto [DecoderName, Decoder] : Decoders) {
    revng_log(Log, "Trying " << DecoderName);
    LoggerIndent Indent(Log);
    if (auto String = Decoder(Data, ProcessCodePoint);
        String.isValid() and String.codePointCount() > 4) {
      revng_log(Log, "Decoding successful!");
      return String;
    }
  }

  return {};
}

UnicodeCStringView UnicodeCStringView::getPrintable(ArrayRef<uint8_t> Data) {
  auto IsPrintable =
    [](size_t Offset, size_t CodePointIndex, uint32_t CodePoint) {
      using namespace llvm::sys::unicode;
      const uint32_t BEL = 0x0007;
      const uint32_t BS = 0x0008;
      const uint32_t HT = 0x0009;
      const uint32_t LF = 0x000A;
      const uint32_t VT = 0x000B;
      const uint32_t FF = 0x000C;
      const uint32_t CR = 0x000D;
      return isPrintable(CodePoint) or CodePoint == BEL or CodePoint == BS
             or CodePoint == HT or CodePoint == LF or CodePoint == VT
             or CodePoint == FF or CodePoint == CR;
    };
  return get(Data, IsPrintable);
}
