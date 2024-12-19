#pragma once

#include <cstdint>
#include <map>
#include <optional>

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace mlir::clift {

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
};

} // namespace mlir::clift
