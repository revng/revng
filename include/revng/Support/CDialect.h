#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Hashing.h"

/// Describes the set of features supported by a dialect of C.
/// \note See `revng/Support/CDialect.inc` for the list of options.
struct CDialect {
  static const CDialect Default;

#define C_DIALECT_OPTION(Option) bool Option = false;
#include "revng/Support/CDialect.inc"

  [[nodiscard]] friend size_t hash_value(const CDialect &C) {
    size_t HashCode = 0;

#define C_DIALECT_OPTION(Option) \
  HashCode = llvm::hash_combine(HashCode, llvm::hash_code(C.Option));

#include "revng/Support/CDialect.inc"
    return HashCode;
  }

  [[nodiscard]] friend bool operator==(const CDialect &,
                                       const CDialect &) = default;
};

inline constexpr CDialect CDialect::Default;
