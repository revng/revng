#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <bitset>
#include <cstddef>
#include <cstdint>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Debug.h"

namespace vma {

/// \brief Index of each color in the bitset
enum ColorIndex : uint8_t {
  POINTERNESS_INDEX,
  UNSIGNEDNESS_INDEX,
  BOOLNESS_INDEX,
  SIGNEDNESS_INDEX,
  FLOATNESS_INDEX,
  NUMBERNESS_INDEX,
  MAX_COLORS
};

/// \brief Useful constants representing one, none and all colors
enum BaseColor : unsigned {
  NO_COLOR = 0,
  POINTERNESS = (1 << POINTERNESS_INDEX),
  UNSIGNEDNESS = (1 << UNSIGNEDNESS_INDEX),
  BOOLNESS = (1 << BOOLNESS_INDEX),
  SIGNEDNESS = (1 << SIGNEDNESS_INDEX),
  FLOATNESS = (1 << FLOATNESS_INDEX),
  NUMBERNESS = (1 << NUMBERNESS_INDEX),
  ALL_COLORS = (1 << MAX_COLORS) - 1
};

/// \brief Name of each color, used for printing
const llvm::StringRef TypeColorName[] = { "P", "U", "B", "S", "F", "N" };

/// \brief Set of colors, stored as a bitset
struct ColorSet {
  using BitsetT = std::bitset<MAX_COLORS>;
  BitsetT Bits;

  ColorSet() = default;
  ~ColorSet() = default;
  ColorSet(const ColorSet &) = default;
  ColorSet(ColorSet &&) = default;
  ColorSet &operator=(const ColorSet &) = default;
  ColorSet &operator=(ColorSet &&) = default;

  ColorSet(const unsigned U) : Bits(U) {}
  ColorSet(const BitsetT B) : Bits(B) {}
  ColorSet(const BitsetT &B) : Bits(B) {}
  ColorSet(BitsetT &&B) : Bits(B) {}

  /// \brief Check if this ColorSet contains all colors of the argument
  bool contains(const ColorSet &Other) const {
    return (Bits | Other.Bits) == Bits;
  }

  /// \brief Add all colors of the argument to this ColorSet
  void addColor(const ColorSet &Other) { Bits |= Other.Bits; }

  /// \brief Count how many valid type candidates are contained in this ColorSet
  size_t countValid() const { return (Bits & BitsetT(~NUMBERNESS)).count(); }

  /// \brief Print the name of the colors contained in this ColorSet
  void print(llvm::raw_ostream &Out) const debug_function {
    if (Bits == ALL_COLORS) {
      Out << "all";
      return;
    }

    for (size_t I = 0; I < MAX_COLORS; ++I)
      if (Bits.test(I))
        Out << TypeColorName[I];
  }

  /// \brief Index of the next set bit, starting from \a StartIndex (excluded)
  /// \param Idx  Where to start from (if == -1 start from the first)
  /// \return MAX_COLORS if there's no set bit after this index
  ColorIndex nextSetBit(int StartIndex) const {
    if (StartIndex < 0)
      StartIndex = -1;

    for (int I = StartIndex + 1; I < MAX_COLORS; I++) {
      if (Bits.test(I))
        return ColorIndex(I);
    }

    return MAX_COLORS;
  }

  /// \brief Index of the first set bit
  /// \return MAX_COLORS if there's no set bit
  ColorIndex firstSetBit() const { return nextSetBit(/*StartIndex*/ -1); }

  friend bool operator==(const ColorSet &Lhs, const ColorSet &Rhs) {
    return Lhs.Bits == Rhs.Bits;
  }

  friend bool operator!=(const ColorSet &Lhs, const ColorSet &Rhs) {
    return !(Lhs == Rhs);
  }
};

} // namespace vma