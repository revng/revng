#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <bitset>
#include <compare>
#include <map>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Support/Debug.h"

namespace vma {

/// Index of each color in the bitset
enum ColorIndex : uint8_t {
  POINTERNESS_INDEX,
  UNSIGNEDNESS_INDEX,
  BOOLNESS_INDEX,
  SIGNEDNESS_INDEX,
  FLOATNESS_INDEX,
  NUMBERNESS_INDEX,
  MAX_COLORS
};

/// Useful constants representing one, none and all colors
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

/// Name of each color, used for printing
const llvm::StringRef TypeColorName[] = { "P", "U", "B", "S", "F", "N" };

/// Set of colors, stored as a bitset
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

  /// Check if this ColorSet contains all colors of the argument
  bool contains(const ColorSet &Other) const {
    return (Bits | Other.Bits) == Bits;
  }

  /// Add all colors of the argument to this ColorSet
  void addColor(const ColorSet &Other) { Bits |= Other.Bits; }

  /// Count how many valid type candidates are contained in this ColorSet
  size_t countValid() const { return (Bits & BitsetT(~NUMBERNESS)).count(); }

  /// Print the name of the colors contained in this ColorSet
  void print(llvm::raw_ostream &Out) const debug_function {
    if (Bits == ALL_COLORS) {
      Out << "all";
      return;
    }

    for (size_t I = 0; I < MAX_COLORS; ++I)
      if (Bits.test(I))
        Out << TypeColorName[I];
  }

  /// Index of the next set bit, starting from \a StartIndex (excluded)
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

  /// Index of the first set bit
  /// \return MAX_COLORS if there's no set bit
  ColorIndex firstSetBit() const { return nextSetBit(/*StartIndex*/ -1); }

  friend std::strong_ordering operator<=>(const ColorSet &Lhs,
                                          const ColorSet &Rhs) {
    return Lhs.Bits.to_ulong() <=> Rhs.Bits.to_ulong();
  }

  friend bool operator==(const ColorSet &Lhs, const ColorSet &Rhs) {
    return Lhs.Bits.to_ulong() == Rhs.Bits.to_ulong();
  }
};

// --------------- Model Colors

/// Colors that are in a 1-to-1 relationship with primitive kinds of the model
enum ModelColor : unsigned {
  VOID = NO_COLOR,
  POINTER = POINTERNESS,
  SIGNED = SIGNEDNESS,
  UNSIGNED = UNSIGNEDNESS,
  FLOAT = FLOATNESS,
  NUMBER = SIGNED | UNSIGNED,
  PTR_OR_NUMBER = NUMBER | POINTER,
  GENERIC = PTR_OR_NUMBER | FLOAT
};

inline const ColorSet ModelColors[] = {
  ModelColor::POINTER, ModelColor::SIGNED, ModelColor::UNSIGNED,
  ModelColor::FLOAT,   ModelColor::NUMBER, ModelColor::PTR_OR_NUMBER,
  ModelColor::GENERIC
};

/// Map each model color to a primitive kind
inline const std::map<ColorSet, model::PrimitiveTypeKind::Values>
  ColorToPrimitiveType = {
    { ModelColor::POINTER, model::PrimitiveTypeKind::PointerOrNumber },
    { ModelColor::SIGNED, model::PrimitiveTypeKind::Signed },
    { ModelColor::UNSIGNED, model::PrimitiveTypeKind::Unsigned },
    { ModelColor::FLOAT, model::PrimitiveTypeKind::Float },
    { ModelColor::NUMBER, model::PrimitiveTypeKind::Number },
    { ModelColor::PTR_OR_NUMBER, model::PrimitiveTypeKind::PointerOrNumber },
    { ModelColor::GENERIC, model::PrimitiveTypeKind::Generic },
  };

/// Map each primitive kind to a model color
inline const std::map<model::PrimitiveTypeKind::Values, ModelColor>
  PrimitiveTypeToColor = {
    { model::PrimitiveTypeKind::Signed, ModelColor::SIGNED },
    { model::PrimitiveTypeKind::Unsigned, ModelColor::UNSIGNED },
    { model::PrimitiveTypeKind::Float, ModelColor::FLOAT },
    { model::PrimitiveTypeKind::Number, ModelColor::NUMBER },
    { model::PrimitiveTypeKind::PointerOrNumber, ModelColor::PTR_OR_NUMBER },
    { model::PrimitiveTypeKind::Generic, ModelColor::GENERIC },
  };

/// Get the smallest ModelColor that contains the given ColorSet
inline ColorSet getNearestModelColor(ColorSet Color) {
  ColorSet BestModelColor = NO_COLOR;
  for (auto MC : ModelColors)
    if (MC.contains(Color))
      if (BestModelColor == NO_COLOR
          or BestModelColor.countValid() > MC.countValid())
        BestModelColor = MC;

  return BestModelColor;
}

} // namespace vma
