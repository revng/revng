#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "revng/Support/Assert.h"

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

enum class CIntegerKind : uint8_t {
  Char,
  Short,
  Int,
  Long,
  LongLong,
  Extended,
};

struct TargetCImplementation {
  uint8_t PointerSize;
  std::map<uint8_t, CIntegerKind> IntegerTypes;

  [[nodiscard]] std::optional<CIntegerKind>
  getIntegerKind(uint64_t const IntegerSize) const {
    auto It = IntegerTypes.find(IntegerSize);
    if (It != IntegerTypes.end())
      return It->second;
    return std::nullopt;
  }

  [[nodiscard]] uint64_t getIntSize() const {
    for (auto [Size, Kind] : IntegerTypes) {
      if (Kind == CIntegerKind::Int)
        return Size;
    }
    revng_abort("Size of CIntegerKind::Int not specified");
  }

  static const TargetCImplementation Default;
};
