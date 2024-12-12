/// \file NameBuilder.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"

static size_t trailingDigitCount(llvm::StringRef Name) {
  size_t Result = 0;
  for (auto Character : std::views::reverse(Name)) {
    if (not std::isdigit(Character))
      return Result;
    else
      ++Result;
  }

  revng_assert(Result == Name.size());
  return Result;
}

const std::set<llvm::StringRef> ReservedKeywords = {
  // Reserved keywords for primitive types.
  // Note that the primitive types we emit are checked separately.
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

static bool isPrefixAndIndex(llvm::StringRef Name, llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  size_t DigitCount = trailingDigitCount(Name);
  return DigitCount != 0 && Name.size() - Prefix.size() == DigitCount;
}

static bool isPrefixAndAddress(llvm::StringRef Name, llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  // This is a hack to let us reuse meta-address parsing.
  //
  // We want this because this would allow keeping names like
  // `function_<valid_address>_anything` while still banning
  // `function_<valid_address>`.
  //
  // TODO: find a cleaner solution for detecting dehydrated (`:` -> `_`)
  //       addresses.
  std::string RehydratedAddress = Name.substr(Prefix.size()).str();
  for (size_t Index = 0; Index < RehydratedAddress.size(); ++Index) {
    if (RehydratedAddress[Index] != '_')
      continue;

    if (Index > 5) {
      // Note that this specifies "Code_" because currently that's the only
      // place where `MetaAddressType` can contain a `_`.
      if (RehydratedAddress.substr(Index - 4, 5) == "Code_") {
        auto Remainder = llvm::StringRef(RehydratedAddress).substr(Index - 4);
        size_t S = MetaAddressType::consumeFromString(Remainder).size();
        if (S != RehydratedAddress.size()) {
          revng_assert(RehydratedAddress.size() >= S);
          Index += RehydratedAddress.size() - S - 1;
          continue;
        }
      }
    }

    RehydratedAddress[Index] = ':';
  }

  return MetaAddress::fromString(RehydratedAddress).isValid();
}

static bool isPrefixAndTypeDefinitionKey(llvm::StringRef Name,
                                         llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  llvm::StringRef WithoutPrefix = Name.substr(Prefix.size());

  namespace TDK = model::TypeDefinitionKind;
  llvm::StringRef WithoutKind = TDK::consumeNamePrefix(WithoutPrefix);
  if (WithoutPrefix.size() == WithoutKind.size())
    return false;

  return isPrefixAndIndex(WithoutKind, "");
}

static bool isPrefixAndRegister(llvm::StringRef Name, llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  llvm::StringRef WithoutPrefix = Name.substr(Prefix.size());

  for (model::Register::Values R : model::Register::allRegisters())
    if (WithoutPrefix == model::Register::getRegisterName(R))
      return true;

  return false;
}

using NC = model::NamingConfiguration;
bool model::NameBuilder::isNameReserved(llvm::StringRef Name,
                                        const NC &Configuration) {
  revng_assert(!Name.empty());

  // Only alphanumeric characters and '_' are allowed
  auto isCharacterForbidden = [](char Character) -> bool {
    return Character != '_' && !std::isalnum(Character);
  };
  if (revng::any_of(Name, isCharacterForbidden))
    return true;

  if (std::isdigit(Name[0]))
    return true;

  // Filter out primitive names we use - we don't want collisions with those
  if (model::PrimitiveType::isCName(Name))
    return true;

  // Forbid names reserved by the C language and some common extensions.
  if (ReservedKeywords.contains(Name))
    return true;

  //
  // The following forbidden names are based on the configuration
  //

  if (Configuration.ReserveNamesStartingWithUnderscore())
    if (Name[0] == '_')
      return true;

  // Prefix + `[0-9]+`
  if (isPrefixAndIndex(Name, Configuration.unnamedSegmentPrefix()))
    return true;
  if (isPrefixAndIndex(Name, Configuration.unnamedStructFieldPrefix()))
    return true;
  if (isPrefixAndIndex(Name, Configuration.unnamedUnionFieldPrefix()))
    return true;
  if (isPrefixAndIndex(Name, Configuration.unnamedFunctionArgumentPrefix()))
    return true;

  // NOTE: This should live in the "Prefix + `[0-9]+`" section, but because we
  //       parse these names when importing from C, let's be extra cautious and
  //       forbid them all.
  if (Name.starts_with(Configuration.structPaddingPrefix()))
    return true;

  // Prefix + MetaAddress.toIdentifier()
  if (isPrefixAndAddress(Name, Configuration.unnamedFunctionPrefix()))
    return true;

  // Prefix + toString(TypeDefinitionKey)
  if (isPrefixAndTypeDefinitionKey(Name,
                                   Configuration.unnamedTypeDefinitionPrefix()))
    return true;

  // NOTE: since automatic enum entry names depend on (potentially unreserved)
  //       enum names, we have no choice but to ban everything starting with
  //       this prefix that also ends with a number.
  if (Name.starts_with(Configuration.unnamedEnumEntryPrefix())
      and std::isdigit(Name.back()))
    return true;

  // Prefix + model::Register::getRegisterName(Register)
  if (isPrefixAndRegister(Name, Configuration.unnamedFunctionRegisterPrefix()))
    return true;

  // NOTE: since artificial return value struct name depends on a (potentially
  //       unreserved) function type name, we have no choice but to ban
  //       everything starting with this prefix.
  if (Name.starts_with(Configuration.artificialReturnValuePrefix()))
    return true;

  // NOTE: since these names are kind of complicated to produce (they recur
  //       on the array's element type), forbid them all.
  if (Name.starts_with(Configuration.artificialArrayWrapperPrefix()))
    return true;

  // Exact names
  if (Name == Configuration.artificialArrayWrapperFieldName())
    return true;

  // Anything else to add here?

  return false;
}

using T = model::Type;
RecursiveCoroutine<std::string>
model::NameBuilder::artificialArrayWrapperNameImpl(const T &Type) {
  if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    std::string Result = "array_" + std::to_string(Array->ElementCount())
                         + "_of_";
    Result += rc_recur artificialArrayWrapperNameImpl(*Array->ElementType());
    rc_return Result;

  } else if (auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return Result += this->name(D->unwrap());

  } else if (auto *Pointer = llvm::dyn_cast<model::PointerType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_ptr_to_" : "ptr_to_");
    Result += rc_recur artificialArrayWrapperNameImpl(*Pointer->PointeeType());
    rc_return Result;

  } else if (auto *Primitive = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    std::string Result = (D->IsConst() ? "const_" : "");
    rc_return Result += Primitive->getCName();

  } else {
    revng_abort("Unsupported model::Type.");
  }
}
