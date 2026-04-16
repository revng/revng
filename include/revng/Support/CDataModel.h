#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Support/Debug.h"

enum class CStandardType : uint8_t {
  Char,
  Short,
  Int,
  Long,
  LongLong,

  Float,
  Double,
  LongDouble,

  Count
};

inline constexpr llvm::StringRef CStandardTypeName[] = {
  "char", "short", "int", "long", "long long", "float", "double", "long double",
};

[[nodiscard]] inline constexpr bool isIntegerType(CStandardType T) {
  return CStandardType::Char <= T and T <= CStandardType::LongLong;
}

[[nodiscard]] inline constexpr bool isFloatingPointType(CStandardType T) {
  return CStandardType::Float <= T and T <= CStandardType::LongDouble;
}

[[nodiscard]] inline std::optional<CStandardType>
findCStandardType(llvm::StringRef Name) {
  auto B = std::begin(CStandardTypeName);
  auto E = std::end(CStandardTypeName);

  auto I = std::find(B, E, Name);
  if (I != E)
    return static_cast<CStandardType>(I - B);

  return std::nullopt;
}

class CDataModel {
public:
  uint8_t PointerSize = 0;
  uint8_t StandardTypeSize[static_cast<uint8_t>(CStandardType::Count)] = {};

  /// Returns the default data model for a given pointer size.
  ///
  /// The default data model is:
  /// * char .......... 1
  /// * short ......... 2
  /// * int ........... 2 if pointer size is less than or equal to 2, or
  ///                   4 otherwise
  /// * long .......... 4 if pointer size is less than or equal to 4, or
  ///                   8 otherwise
  /// * long long ..... 8
  /// * float ......... 4
  /// * double ........ 8
  /// * long double ... 8
  [[nodiscard]] static CDataModel getDefaultDataModel(uint8_t PointerSize);

  [[nodiscard]] unsigned getPointerSize() const { return PointerSize; };

  [[nodiscard]] uint8_t &getStandardTypeSize(CStandardType T) {
    return StandardTypeSize[static_cast<uint8_t>(T)];
  }

  [[nodiscard]] const uint8_t &getStandardTypeSize(CStandardType T) const {
    return StandardTypeSize[static_cast<uint8_t>(T)];
  }

  void setStandardTypeSize(CStandardType T, uint8_t Size) & {
    getStandardTypeSize(T) = Size;
  }

  [[nodiscard]] unsigned getCharSize() const {
    return getStandardTypeSize(CStandardType::Char);
  }

  [[nodiscard]] unsigned getShortSize() const {
    return getStandardTypeSize(CStandardType::Short);
  }

  [[nodiscard]] unsigned getIntSize() const {
    return getStandardTypeSize(CStandardType::Int);
  }

  [[nodiscard]] unsigned getLongSize() const {
    return getStandardTypeSize(CStandardType::Long);
  }

  [[nodiscard]] unsigned getLongLongSize() const {
    return getStandardTypeSize(CStandardType::LongLong);
  }

  [[nodiscard]] unsigned getFloatSize() const {
    return getStandardTypeSize(CStandardType::Float);
  }

  [[nodiscard]] unsigned getDoubleSize() const {
    return getStandardTypeSize(CStandardType::Double);
  }

  [[nodiscard]] unsigned getLongDoubleSize() const {
    return getStandardTypeSize(CStandardType::LongDouble);
  }

  /// Returns the (inclusive) range of standard integer types matching the
  /// specified size, or nullopt if no such integer types are available.
  [[nodiscard]] std::optional<std::pair<CStandardType, CStandardType>>
  getStandardIntegerRange(uint8_t Size) const {
    CStandardType Min = CStandardType::Char;
    while (Min <= CStandardType::LongLong and getStandardTypeSize(Min) < Size)
      Min = next(Min);

    if (Min > CStandardType::LongLong)
      return std::nullopt;

    CStandardType Max = Min;
    while (Max < CStandardType::LongLong
           and getStandardTypeSize(next(Max)) <= Size)
      Max = next(Max);

    return std::pair<CStandardType, CStandardType>(Min, Max);
  }

  [[nodiscard]] bool verify() const;
  void dump() const debug_function;

  [[nodiscard]] friend size_t hash_value(const CDataModel &DM) {
    size_t HashCode = llvm::hash_code(DM.PointerSize);

    for (int I = 0; I < static_cast<int>(CStandardType::Count); ++I)
      HashCode = llvm::hash_combine(llvm::hash_code(DM.StandardTypeSize[I]));

    return HashCode;
  }

  [[nodiscard]] friend bool operator==(const CDataModel &,
                                       const CDataModel &) = default;

private:
  static CStandardType next(CStandardType T) {
    return static_cast<CStandardType>(static_cast<uint8_t>(T) + 1);
  }
};
