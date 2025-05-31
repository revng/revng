/// \file NameBuilder.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Intrinsics.h"

#include "revng/Model/Binary.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Support/Annotations.h"
#include "revng/Support/Identifier.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/YAMLTraits.h"

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

static llvm::StringRef dropWrapperSuffix(llvm::StringRef Name) {
  size_t Position = Name.find("_wrapper");
  if (Position != llvm::StringRef::npos)
    return Name.substr(0, Position);

  return Name;
}

static bool isLLVMIntrinsic(llvm::StringRef Name) {
  // Since intrinsics are all guaranteed to start with `llvm_`,
  // we can skip the check for most of the identifiers.
  if (not Name.starts_with("llvm_"))
    return false;

  // Start iteration from 1 because 0 is reserved for `not_intrinsic`.
  for (llvm::Intrinsic::ID I = 1; I < llvm::Intrinsic::num_intrinsics; ++I) {
    auto Sanitized = sanitizeIdentifier(llvm::Intrinsic::getBaseName(I));
    if (Name.starts_with(Sanitized))
      return true;
  }

  return false;
}

static const SortedVector<std::string> &loadHelperNameList() {
  static std::optional<SortedVector<std::string>> Cache = std::nullopt;
  if (not Cache.has_value()) {
    auto MaybePath = revng::ResourceFinder.findFile("share/revng/"
                                                    "helper-list.csv");
    revng_assert(MaybePath.has_value(),
                 ("Helper list is missing: " + *MaybePath).c_str());

    auto MaybeBuffer = llvm::MemoryBuffer::getFile(*MaybePath, true);
    revng_assert(MaybeBuffer, ("Can't open " + *MaybePath).c_str());

    llvm::SmallVector<llvm::StringRef, 8> Names;
    llvm::StringRef(MaybeBuffer->get()->getBuffer()).split(Names, "\n");

    revng_assert(!Names.empty());
    revng_assert(Names[0] == "Name");
    revng_assert(Names.back().empty());
    Names.pop_back();

    Cache = SortedVector<std::string>();
    auto Inserter = Cache->batch_insert();
    for (llvm::StringRef Name : Names | std::views::drop(1)) {
      revng_assert(not Name.empty());
      revng_assert(Name == Name.trim());

      // We don't want to ban `main`, even if it's is in the helper module.
      if (Name == "main")
        continue;

      // We don't care about llvm debug information.
      if (Name.starts_with("llvm.dbg"))
        continue;

      // No reason to double-ban intrinsics.
      if (isLLVMIntrinsic(Name))
        continue;

      // TODO: other names we don't want to ban?

      Inserter.insert(sanitizeIdentifier(Name));
    }
  }

  return *Cache;
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

/// Returns `true` iff the identifier is exactly the given prefix + a decimal
/// number. For example (Prefix = `argument_` ):
/// * `argument_4` - `true`
/// * `argument_4a` - `false`
/// * `argument_4_` - `false`
/// * `argument_` - `false`
static bool isPrefixAndIndex(llvm::StringRef Name, llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  size_t DigitCount = trailingDigitCount(Name);
  return DigitCount != 0 && Name.size() - Prefix.size() == DigitCount;
}

/// Returns `true` iff the identifier is exactly the given prefix +
/// a serialized (flattened) `MetaAddress`. For example (Prefix = `function_`):
/// * `function_0x1004_Code_arm` - `true`
/// * `function_0x1004_Code_arma` - `false`
/// * `function_0x1004_Code_arm_` - `false`
/// * `function_0x1004_Co_de_arm` - `false`
/// * `function_` - `false`
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

/// Returns `true` iff the identifier is exactly the given prefix +
/// a serialized model type key.
///
/// Note that unlike other similar functions, the prefix is allowed (and even
/// expected) to be empty here.
///
/// For example (Prefix = ``):
/// * `struct_42` - `true`
/// * `struct_42a` - `false`
/// * `struct_42_` - `false`
/// * `struct_4_2` - `false`
/// * `struct_` - `false`
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

/// Returns `true` iff the identifier is exactly the given prefix +
/// a serialized register name *without* the architecture.
///
/// For example (Prefix = `register_`):
/// * `register_r4` - `true`
/// * `register_r4a` - `false`
/// * `register_r4_` - `false`
/// * `register_r_4` - `false`
/// * `register_` - `false`
static bool isPrefixAndRegister(llvm::StringRef Name, llvm::StringRef Prefix) {
  if (not Name.starts_with(Prefix))
    return false;

  llvm::StringRef WithoutPrefix = Name.substr(Prefix.size());

  for (model::Register::Values R : model::Register::allRegisters())
    if (WithoutPrefix == model::Register::getRegisterName(R))
      return true;

  return false;
}

llvm::Error model::CNameBuilder::isNameReserved(llvm::StringRef Name) const {
  revng_assert(!Name.empty());

  // Only alphanumeric characters and '_' are allowed
  // TODO: handle unicode
  auto IsCharacterForbidden = [](char Character) -> bool {
    return Character != '_' && !std::isalnum(Character);
  };
  auto Iterator = llvm::find_if(Name, IsCharacterForbidden);
  if (Iterator != Name.end()) {
    using namespace std::string_literals;
    return revng::createError("it contains a forbidden character: `"s
                              + *Iterator + '`');
  }

  if (std::isdigit(Name[0]))
    return revng::createError("it starts with a digit: `"s + Name[0] + '`');

  // Filter out primitive names we use - we don't want collisions with those
  if (model::PrimitiveType::isCName(Name))
    return revng::createError("it is reserved for a primitive type we use");

  // Forbid names reserved by the C language and some common extensions.
  if (ReservedKeywords.contains(Name))
    return revng::createError("it is reserved by the C language or some common "
                              "extension");

  llvm::StringRef Unwrapped = dropWrapperSuffix(Name);
  if (loadHelperNameList().contains(Unwrapped.str()))
    return revng::createError("it collides with a helper (`" + Unwrapped.str()
                              + "`)");

  if (isLLVMIntrinsic(Name))
    return revng::createError("it collides with one of the LLVM intrinsics");

  //
  // The following names are reserved based on the configuration
  //

  // Names we emit as macros.
  if (ptml::Attributes.isMacro(Name))
    return revng::createError("it is reserved for a macro we use in the "
                              "decompiled code");

  if (Configuration.ReserveNamesStartingWithUnderscore())
    if (Name[0] == '_')
      return revng::createError("it is reserved because it begins with an "
                                "underscore");

  // Prefix + `[0-9]+`
  if (isPrefixAndIndex(Name, Configuration.unnamedSegmentPrefix()))
    return revng::createError("it is reserved for an automatic segment name");
  if (isPrefixAndIndex(Name, Configuration.unnamedStructFieldPrefix()))
    return revng::createError("it is reserved for an automatic struct field "
                              "name");
  if (isPrefixAndIndex(Name, Configuration.unnamedUnionFieldPrefix()))
    return revng::createError("it is reserved for an automatic struct union "
                              "name");
  if (isPrefixAndIndex(Name, Configuration.unnamedFunctionArgumentPrefix()))
    return revng::createError("it is reserved for an automatic function "
                              "argument name");
  if (isPrefixAndIndex(Name, Configuration.unnamedLocalVariablePrefix()))
    return revng::createError("it is reserved for an automatic local variable "
                              "name");
  if (isPrefixAndIndex(Name,
                       Configuration.unnamedBreakFromLoopVariablePrefix()))
    return revng::createError("it is reserved for a break-from-loop variable "
                              "name");

  // NOTE: This should live in the "Prefix + `[0-9]+`" section, but because we
  //       parse these names when importing from C, let's be extra cautious and
  //       forbid them all.
  if (Name.starts_with(Configuration.structPaddingPrefix()))
    return revng::createError("it is reserved for a struct padding name");

  // Prefix + MetaAddress.toIdentifier()
  if (isPrefixAndAddress(Name, Configuration.unnamedFunctionPrefix()))
    return revng::createError("it is reserved for an automatic function name");

  // Prefix + toString(TypeDefinitionKey)
  if (isPrefixAndTypeDefinitionKey(Name,
                                   Configuration.unnamedTypeDefinitionPrefix()))
    return revng::createError("it is reserved for an automatic type name");

  // NOTE: since automatic enum entry names depend on (potentially unreserved)
  //       enum names, we have no choice but to reserve everything starting with
  //       this prefix that also ends with a number.
  if (Name.starts_with(Configuration.unnamedEnumEntryPrefix())
      and std::isdigit(Name.back()))
    return revng::createError("it is reserved for an automatic enum entry "
                              "name");

  // NOTE: since we parse these it's safer to reserve all of them.
  if (Name.starts_with(Configuration.maximumEnumValuePrefix()))
    return revng::createError("it is reserved for a maximum enum value");

  // Prefix + model::Register::getRegisterName(Register)
  if (isPrefixAndRegister(Name, Configuration.unnamedFunctionRegisterPrefix()))
    return revng::createError("it is reserved for an automatic register "
                              "argument name");

  // NOTE: since artificial return value struct name depends on a (potentially
  //       unreserved) function type name, we have no choice but to reserve
  //       everything starting with this prefix.
  if (Name.starts_with(Configuration.artificialReturnValuePrefix()))
    return revng::createError("it is reserved for an artificial return value "
                              "struct name");

  // NOTE: since these names are kind of complicated to produce (they recur
  //       on the array's element type), forbid them all.
  if (Name.starts_with(Configuration.artificialArrayWrapperPrefix()))
    return revng::createError("it is reserved for an artificial array wrapper "
                              "name");

  // TODO: more granularity is possible here since this prefix is only ever
  //       followed by a primitive name, but forbid them all for now.
  if (Name.starts_with(Configuration.undefinedValuePrefix()))
    return revng::createError("it is reserved for an undefined value");

  // NOTE: since CSV value names are kind of external, reserve everything just
  //       to be safe.
  if (Name.starts_with(Configuration.opaqueCSVValuePrefix()))
    return revng::createError("it is reserved for an opaque CSV value");

  // NOTE: since these use a hash suffix, let's be on the safe side and reserve
  //       the entire prefix.
  if (Name.starts_with(Configuration.unnamedDynamicFunctionPrefix()))
    return revng::createError("it is reserved for a dynamic function name");

  //
  // Hardcoded prefixes
  //

  // TODO: We can be more careful with these and only reserve the ones we use,
  //       once `CTargetImplementation` is mature enough to give us the list.
  if (Name.starts_with("__builtin_"))
    return revng::createError("it is reserved for a builtin intrinsic");

  //
  // Exact names
  //

  if (Name == Configuration.artificialArrayWrapperFieldName())
    return revng::createError("it is reserved for an artificial array wrapper "
                              "field name");
  if (Name == Configuration.stackFrameVariableName())
    return revng::createError("it is reserved for a stack variable name");
  if (Name == Configuration.rawStackArgumentName())
    return revng::createError("it is reserved for a stack argument name");
  if (Name == Configuration.loopStateVariableName())
    return revng::createError("it is reserved for a loop state variable name");

  // Anything else to add here?

  return llvm::Error::success();
}

llvm::Error
model::AssemblyNameBuilder::isNameReserved(llvm::StringRef Name) const {
  revng_assert(!Name.empty());

  // Only alphanumeric characters and '_' are allowed
  // TODO: handle unicode
  auto IsCharacterForbidden = [](char Character) -> bool {
    return Character != '_' && !std::isalnum(Character);
  };
  auto Iterator = llvm::find_if(Name, IsCharacterForbidden);
  if (Iterator != Name.end()) {
    using namespace std::string_literals;
    return revng::createError("it contains a forbidden character: `"s
                              + *Iterator + '`');
  }

  if (std::isdigit(Name[0]))
    return revng::createError("it starts with a digit: `"s + Name[0] + '`');

  if (Configuration.ReserveNamesStartingWithUnderscore())
    if (Name[0] == '_')
      return revng::createError("it is reserved because it begins with an "
                                "underscore");

  // Prefix + MetaAddress.toIdentifier()
  if (isPrefixAndAddress(Name, Configuration.unnamedFunctionPrefix()))
    return revng::createError("it is reserved for an automatic function name");

  // NOTE: since these use a hash suffix, let's be on the safe side and reserve
  //       the entire prefix.
  if (Name.starts_with(Configuration.unnamedDynamicFunctionPrefix()))
    return revng::createError("it is reserved for a dynamic function name");

  // Register names
  for (model::Register::Values Register : registers(Architecture))
    if (Name == model::Register::getRegisterName(Register))
      return revng::createError("it collides with a register name");

  // Anything else to add here?

  return llvm::Error::success();
}
