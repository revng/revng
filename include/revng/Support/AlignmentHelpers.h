#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>

#include "llvm/Support/MathExtras.h"

#include "revng/Support/Assert.h"

/// Pads the size of a stack argument up to the next multiple of the register
/// size, with a minimum of `RegisterSize`.
///
/// For ABI stack-passed arguments whose declared size is less than a register,
/// the slot still occupies a full register.
///
/// \note This only accounts for the post-padding (extension).
///       Pre-padding (offset) needs to be taken care of separately.
///
/// \param RealSize    The size of the argument without the padding.
/// \param RegisterSize The size of a register under the given architecture.
///
/// \return The size of the argument with the padding.
inline constexpr uint64_t paddedSizeOnStack(uint64_t RealSize,
                                            uint64_t RegisterSize) {
  revng_assert(llvm::isPowerOf2_64(RegisterSize));
  revng_assert(RealSize != 0, "0-sized stack entries are not supported.");

  if (RealSize <= RegisterSize)
    return RegisterSize;

  RealSize += RegisterSize - 1;
  RealSize &= ~(RegisterSize - 1);

  return RealSize;
}
