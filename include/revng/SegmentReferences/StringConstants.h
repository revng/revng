#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/Unicode.h"

class RawBinaryView;

/// The shape rev.ng gives a string: an array of constant characters, one or two
/// bytes wide. `detect-c-strings` creates them and `emit-string-constants`
/// renders what it finds there as a literal.
///
/// \return 1 for `uint8_t const[]`, 2 for `uint16_t const[]`, 0 for anything
///         else.
unsigned getConstCharArrayElementSize(const model::Type &Type);

/// The string the \p ByteCount bytes at \p Address spell, if they spell one.
///
/// The result is invalid unless every one of those bytes is part of a printable
/// NUL-terminated string of \p CharSize -wide characters, terminator included:
/// a run of printable characters that stops short of the end is not the string
/// the array was meant to hold.
UnicodeCStringView readString(RawBinaryView &BinaryView,
                              const MetaAddress &Address,
                              uint64_t ByteCount,
                              unsigned CharSize);
