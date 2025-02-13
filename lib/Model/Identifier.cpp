/// \file Identifier.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DefinedType.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/PointerType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/VerifyHelper.h"

using namespace model;

const Identifier Identifier::Empty = Identifier("");

const std::set<llvm::StringRef> ReservedKeywords = {
  // reserved keywords for primitive types
  "void",
  "pointer_or_number8_t",
  "pointer_or_number16_t",
  "pointer_or_number32_t",
  "pointer_or_number64_t",
  "pointer_or_number128_t",
  "number8_t",
  "number16_t",
  "number32_t",
  "number64_t",
  "number128_t",
  "generic8_t",
  "generic16_t",
  "generic32_t",
  "generic64_t",
  "generic80_t",
  "generic96_t",
  "generic128_t",
  "int8_t",
  "int16_t",
  "int32_t",
  "int64_t",
  "int128_t",
  "int_fast8_t",
  "int_fast16_t",
  "int_fast32_t",
  "int_fast64_t",
  "int_fast128_t",
  "int_least8_t",
  "int_least16_t",
  "int_least32_t",
  "int_least64_t",
  "int_least128_t",
  "intmax_t",
  "intptr_t",
  "uint8_t",
  "uint16_t",
  "uint32_t",
  "uint64_t",
  "uint128_t",
  "uint_fast8_t",
  "uint_fast16_t",
  "uint_fast32_t",
  "uint_fast64_t",
  "uint_fast128_t",
  "uint_least8_t",
  "uint_least16_t",
  "uint_least32_t",
  "uint_least64_t",
  "uint_least128_t",
  "uintmax_t",
  "uintptr_t",
  "float16_t",
  "float32_t",
  "float64_t",
  "float80_t",
  "float96_t",
  "float128_t",
  // Integer macros from stdint.h, reserved to prevent clashes.
  "INT8_WIDTH",
  "INT16_WIDTH",
  "INT32_WIDTH",
  "INT64_WIDTH",
  "INT_FAST8_WIDTH",
  "INT_FAST16_WIDTH",
  "INT_FAST32_WIDTH",
  "INT_FAST64_WIDTH",
  "INT_LEAST8_WIDTH",
  "INT_LEAST16_WIDTH",
  "INT_LEAST32_WIDTH",
  "INT_LEAST64_WIDTH",
  "INTPTR_WIDTH",
  "INTMAX_WIDTH",
  "INT8_MIN",
  "INT16_MIN",
  "INT32_MIN",
  "INT64_MIN",
  "INT_FAST8_MIN",
  "INT_FAST16_MIN",
  "INT_FAST32_MIN",
  "INT_FAST64_MIN",
  "INT_LEAST8_MIN",
  "INT_LEAST16_MIN",
  "INT_LEAST32_MIN",
  "INT_LEAST64_MIN",
  "INTPTR_MIN",
  "INTMAX_MIN",
  "INT8_MAX",
  "INT16_MAX",
  "INT32_MAX",
  "INT64_MAX",
  "INT_FAST8_MAX",
  "INT_FAST16_MAX",
  "INT_FAST32_MAX",
  "INT_FAST64_MAX",
  "INT_LEAST8_MAX",
  "INT_LEAST16_MAX",
  "INT_LEAST32_MAX",
  "INT_LEAST64_MAX",
  "INTPTR_MAX",
  "INTMAX_MAX",
  "UINT8_WIDTH",
  "UINT16_WIDTH",
  "UINT32_WIDTH",
  "UINT64_WIDTH",
  "UINT_FAST8_WIDTH",
  "UINT_FAST16_WIDTH",
  "UINT_FAST32_WIDTH",
  "UINT_FAST64_WIDTH",
  "UINT_LEAST8_WIDTH",
  "UINT_LEAST16_WIDTH",
  "UINT_LEAST32_WIDTH",
  "UINT_LEAST64_WIDTH",
  "UINTPTR_WIDTH",
  "UINTMAX_WIDTH",
  "UINT8_MAX",
  "UINT16_MAX",
  "UINT32_MAX",
  "UINT64_MAX",
  "UINT_FAST8_MAX",
  "UINT_FAST16_MAX",
  "UINT_FAST32_MAX",
  "UINT_FAST64_MAX",
  "UINT_LEAST8_MAX",
  "UINT_LEAST16_MAX",
  "UINT_LEAST32_MAX",
  "UINT_LEAST64_MAX",
  "UINTPTR_MAX",
  "UINTMAX_MAX",
  "INT8_C",
  "INT16_C",
  "INT32_C",
  "INT64_C",
  "INTMAX_C",
  "UINT8_C",
  "UINT16_C",
  "UINT32_C",
  "UINT64_C",
  "UINTMAX_C",
  // C reserved keywords
  "auto",
  "break",
  "case",
  "char",
  "const",
  "continue",
  "default",
  "do",
  "double",
  "else",
  "enum",
  "extern",
  "float",
  "for",
  "goto",
  "if",
  "inline", // Since C99
  "int",
  "long",
  "register",
  "restrict", // Since C99
  "return",
  "short",
  "signed",
  "sizeof",
  "static",
  "struct",
  "switch",
  "typedef",
  "union",
  "unsigned",
  "volatile",
  "while",
  "_Alignas", // Since C11
  "_Alignof", // Since C11
  "_Atomic", // Since C11
  "_Bool", // Since C99
  "_Complex", // Since C99
  "_Decimal128", // Since C23
  "_Decimal32", // Since C23
  "_Decimal64", // Since C23
  "_Generic", // Since C11
  "_Imaginary", // Since C99
  "_Noreturn", // Since C11
  "_Static_assert", // Since C11
  "_Thread_local", // Since C11
  // Convenience macros
  "alignas",
  "alignof",
  "bool",
  "complex",
  "imaginary",
  "noreturn",
  "static_assert",
  "thread_local",
  // Convenience macros for atomic types
  "atomic_bool",
  "atomic_char",
  "atomic_schar",
  "atomic_uchar",
  "atomic_short",
  "atomic_ushort",
  "atomic_int",
  "atomic_uint",
  "atomic_long",
  "atomic_ulong",
  "atomic_llong",
  "atomic_ullong",
  "atomic_char16_t",
  "atomic_char32_t",
  "atomic_wchar_t",
  "atomic_int_least8_t",
  "atomic_uint_least8_t",
  "atomic_int_least16_t",
  "atomic_uint_least16_t",
  "atomic_int_least32_t",
  "atomic_uint_least32_t",
  "atomic_int_least64_t",
  "atomic_uint_least64_t",
  "atomic_int_fast8_t",
  "atomic_uint_fast8_t",
  "atomic_int_fast16_t",
  "atomic_uint_fast16_t",
  "atomic_int_fast32_t",
  "atomic_uint_fast32_t",
  "atomic_int_fast64_t",
  "atomic_uint_fast64_t",
  "atomic_intptr_t",
  "atomic_uintptr_t",
  "atomic_size_t",
  "atomic_ptrdiff_t",
  "atomic_intmax_t",
  "atomic_uintmax_t",
  // C Extensions
  "_Pragma",
  "asm",
};

static bool isNotUnderscore(const char C) {
  return C != '_';
};

static bool allAlphaNumOrUnderscore(const Identifier &Range) {
  const auto &FilterRange = llvm::make_filter_range(Range, isNotUnderscore);
  for (const auto &Entry : FilterRange)
    if (not std::isalnum(Entry))
      return false;
  return true;
};

bool Identifier::verify(VerifyHelper &VH) const {
  return VH.maybeFail(not(not empty() and std::isdigit(str()[0]))
                        and not startswith("_")
                        and allAlphaNumOrUnderscore(str())
                        and not ReservedKeywords.contains(str()),
                      llvm::Twine(*this) + " is not a valid identifier");
}

bool Identifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}
bool Identifier::verify() const {
  return verify(false);
}

constexpr static llvm::StringRef PrefixForReservedNames = "unreserved_";

Identifier Identifier::fromString(llvm::StringRef Name) {
  revng_assert(not Name.empty());
  Identifier Result;

  // For reserved C keywords prepend a non-reserved prefix and we're done.
  if (ReservedKeywords.contains(Name)) {
    Result += PrefixForReservedNames;
    Result += Name;
    return Result;
  }

  const auto BecomesUnderscore = [](const char C) {
    return not std::isalnum(C) or C == '_';
  };

  // For invalid C identifiers prepend the our reserved prefix.
  if (std::isdigit(Name[0]) or BecomesUnderscore(Name[0])) {
    Result += PrefixForReservedNames;
  }

  // Append the rest of the name
  Result += Name;

  return sanitize(Result);
}

Identifier Identifier::sanitize(llvm::StringRef Name) {
  Identifier Result(Name);

  // Convert all non-alphanumeric chars to underscores
  for (char &C : Result)
    if (not std::isalnum(C))
      C = '_';

  return Result;
}
