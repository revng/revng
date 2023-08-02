//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>
#include <cctype>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Model/VerifyTypeHelper.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::Twine;

namespace model {

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

model::Type::Type() : model::Type(0, model::TypeKind::Invalid){};

model::Type::Type(uint64_t ID, TypeKind::Values Kind) :
  model::generated::Type(ID, Kind) {
}

const llvm::SmallVector<model::QualifiedType, 4> model::Type::edges() const {
  const auto *This = this;
  auto GetEdges = [](const auto &Upcasted) { return Upcasted.edges(); };
  return upcast(This, GetEdges, llvm::SmallVector<model::QualifiedType, 4>());
}

template<size_t I = 0>
model::UpcastableType
makeTypeWithIDImpl(uint64_t ID, model::TypeKind::Values Kind) {
  using concrete_types = concrete_types_traits_t<model::Type>;
  if constexpr (I < std::tuple_size_v<concrete_types>) {
    using type = std::tuple_element_t<I, concrete_types>;
    if (type::classof(typename type::Key(ID, Kind)))
      return UpcastableType(new type(ID, type::AssociatedKind));
    else
      return model::makeTypeWithIDImpl<I + 1>(ID, Kind);
  } else {
    return UpcastableType(nullptr);
  }
}

model::UpcastableType makeTypeWithID(uint64_t ID,
                                     model::TypeKind::Values Kind) {
  return makeTypeWithIDImpl(ID, Kind);
}

Identifier model::UnionField::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (Twine("_member") + Twine(Index())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

Identifier model::StructField::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (Twine("_offset_") + Twine(Offset())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

Identifier model::Argument::name() const {
  Identifier Result;

  if (CustomName().empty()) {
    (Twine("_argument") + Twine(Index())).toVector(Result);
  } else {
    Result = CustomName();
  }

  return Result;
}

Identifier model::Type::name() const {
  auto *This = this;
  auto GetName = [](auto &Upcasted) -> Identifier { return Upcasted.name(); };
  return upcast(This, GetName, Identifier(""));
}

void Qualifier::dump() const {
  serialize(dbg, *this);
}

bool Qualifier::verify() const {
  return verify(false);
}

bool Qualifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Qualifier::verify(VerifyHelper &VH) const {
  switch (Kind()) {
  case QualifierKind::Invalid:
    return VH.fail("Invalid qualifier found", *this);
  case QualifierKind::Pointer:
    return VH.maybeFail(Size() > 0 and llvm::isPowerOf2_64(Size()),
                        "Pointer qualifier size is not a power of 2",
                        *this);
  case QualifierKind::Const:
    return VH.maybeFail(Size() == 0, "const qualifier has non-0 size", *this);
  case QualifierKind::Array:
    return VH.maybeFail(Size() > 0, "Array qualifier size is 0");
  default:
    revng_abort();
  }

  return VH.fail();
}

static constexpr bool isValidPrimitiveSize(PrimitiveTypeKind::Values PrimKind,
                                           uint8_t BS) {
  switch (PrimKind) {
  case PrimitiveTypeKind::Invalid:
    return false;

  case PrimitiveTypeKind::Void:
    return BS == 0;

  // The ByteSizes allowed for Generic must be a superset of all the other
  // ByteSizes allowed for all other primitive types (except void)
  case PrimitiveTypeKind::Generic:
    return BS == 1 or BS == 2 or BS == 4 or BS == 8 or BS == 10 or BS == 12
           or BS == 16;

  case PrimitiveTypeKind::PointerOrNumber:
  case PrimitiveTypeKind::Number:
  case PrimitiveTypeKind::Unsigned:
  case PrimitiveTypeKind::Signed:
    return BS == 1 or BS == 2 or BS == 4 or BS == 8 or BS == 16;

  // NOTE: We are supporting floats that are 10 bytes long, since we found such
  // cases in some PDB files by using VS on Windows platforms. The source code
  // of those cases could be written in some language other than C/C++ (probably
  // Swift). We faced some struct fields by using this (10b long float) type, so
  // by ignoring it we would not have accurate layout for the structs.
  case PrimitiveTypeKind::Float:
    return BS == 2 or BS == 4 or BS == 8 or BS == 10 or BS == 12 or BS == 16;

  default:
    revng_abort();
  }

  revng_abort();
}

Identifier model::PrimitiveType::name() const {
  Identifier Result;

  switch (PrimitiveKind()) {
  case PrimitiveTypeKind::Void:
    Result = "void";
    break;

  case PrimitiveTypeKind::Unsigned:
    (Twine("uint") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Number:
    (Twine("number") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::PointerOrNumber:
    ("pointer_or_number" + Twine(Size() * 8) + "_t").toVector(Result);
    break;

  case PrimitiveTypeKind::Generic:
    (Twine("generic") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Signed:
    (Twine("int") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Float:
    (Twine("float") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  default:
    revng_abort();
  }

  return Result;
}

template<typename T>
Identifier customNameOrAutomatic(T *This) {
  if (not This->CustomName().empty())
    return This->CustomName();
  else {
    auto IdentText = (Twine("_") + Twine(T::AutomaticNamePrefix)
                      + Twine(This->ID()))
                       .str();
    return Identifier(IdentText);
  }
}

Identifier model::StructType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::TypedefType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::EnumType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::EnumType::entryName(const model::EnumEntry &Entry) const {
  revng_assert(Entries().count(Entry.Value()) != 0);

  if (Entry.CustomName().size() > 0) {
    return Entry.CustomName();
  } else {
    return Identifier((Twine("_enum_entry_") + name().str() + "_"
                       + Twine(Entry.Value()))
                        .str());
  }
}

Identifier model::UnionType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::NamedTypedRegister::name() const {
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    using namespace model::Register;
    return Identifier((Twine("_argument_") + Twine(getRegisterName(Location())))
                        .str());
  }
}

Identifier model::RawFunctionType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::CABIFunctionType::name() const {
  return customNameOrAutomatic(this);
}

static uint64_t makePrimitiveID(PrimitiveTypeKind::Values PrimitiveKind,
                                uint8_t Size) {
  return (static_cast<uint8_t>(PrimitiveKind) << 8) | Size;
}

PrimitiveType::PrimitiveType(PrimitiveTypeKind::Values PrimitiveKind,
                             uint8_t Size) :
  PrimitiveType(makePrimitiveID(PrimitiveKind, Size),
                AssociatedKind,
                {},
                {},
                {},
                PrimitiveKind,
                Size) {
}

static PrimitiveTypeKind::Values getPrimitiveKind(uint64_t ID) {
  return static_cast<PrimitiveTypeKind::Values>(ID >> 8);
}

static uint8_t getPrimitiveSize(uint64_t ID) {
  return ID & ((1 << 8) - 1);
}

PrimitiveType::PrimitiveType(uint64_t ID) :
  PrimitiveType(ID,
                AssociatedKind,
                {},
                {},
                {},
                getPrimitiveKind(ID),
                getPrimitiveSize(ID)) {
}

void EnumEntry::dump() const {
  serialize(dbg, *this);
}

bool EnumEntry::verify() const {
  return verify(false);
}

bool EnumEntry::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool EnumEntry::verify(VerifyHelper &VH) const {
  return VH.maybeFail(CustomName().verify(VH));
}

std::optional<uint64_t> QualifiedType::size() const {
  VerifyHelper VH;
  return size(VH);
}

std::optional<uint64_t> QualifiedType::trySize() const {
  VerifyHelper VH;
  return trySize(VH);
}

RecursiveCoroutine<std::optional<uint64_t>>
QualifiedType::size(VerifyHelper &VH) const {
  std::optional<uint64_t> MaybeSize = rc_recur trySize(VH);
  revng_check(MaybeSize);
  if (*MaybeSize == 0)
    rc_return std::nullopt;
  else
    rc_return MaybeSize;
}

RecursiveCoroutine<std::optional<uint64_t>>
QualifiedType::trySize(VerifyHelper &VH) const {
  // This code assumes that the QualifiedType QT is well formed.
  auto QIt = Qualifiers().begin();
  auto QEnd = Qualifiers().end();

  for (; QIt != QEnd; ++QIt) {

    auto &Q = *QIt;
    switch (Q.Kind()) {

    case QualifierKind::Invalid:
      rc_return std::nullopt;

    case QualifierKind::Pointer:
      // If we find a pointer, we're done
      rc_return Q.Size();

    case QualifierKind::Array: {
      // The size is equal to (number of elements of the array) * (size of a
      // single element).
      const QualifiedType ArrayElem{ UnqualifiedType(),
                                     { std::next(QIt), QEnd } };
      auto MaybeSize = rc_recur ArrayElem.trySize(VH);
      if (not MaybeSize)
        rc_return std::nullopt;
      else
        rc_return *MaybeSize *Q.Size();
    }

    case QualifierKind::Const:
      // Do nothing, just skip over it
      break;

    default:
      revng_abort();
    }
  }

  if (!UnqualifiedType().isValid())
    rc_return std::nullopt;

  rc_return rc_recur UnqualifiedType().get()->trySize(VH);
}

static RecursiveCoroutine<bool> isArrayImpl(const model::QualifiedType &QT) {
  const auto &NotIsConst = std::not_fn(model::Qualifier::isConst);
  for (const auto &Q : llvm::make_filter_range(QT.Qualifiers(), NotIsConst)) {

    // If we find an array first, it's definitely an array, otherwise we
    // found a pointer first, so it's definitely not an array
    if (Qualifier::isArray(Q))
      rc_return true;

    rc_return false;
  }

  if (auto *TD = dyn_cast<model::TypedefType>(QT.UnqualifiedType().get()))
    rc_return rc_recur isArrayImpl(TD->UnderlyingType());

  // If there are no non-const qualifiers, it's not an array
  rc_return false;
}

bool QualifiedType::isArray() const {
  return isArrayImpl(*this);
}

static RecursiveCoroutine<bool> isPointerImpl(const model::QualifiedType &QT) {
  const auto &NotIsConst = std::not_fn(Qualifier::isConst);
  for (const auto &Q : llvm::make_filter_range(QT.Qualifiers(), NotIsConst)) {

    // If we find a pointer first, it's definitely a pointer, otherwise we
    // found an array first, so it's definitely not a pointer
    if (Qualifier::isPointer(Q))
      rc_return true;

    rc_return false;
  }

  if (auto *TD = dyn_cast<model::TypedefType>(QT.UnqualifiedType().get()))
    rc_return rc_recur isPointerImpl(TD->UnderlyingType());

  // If there are no non-const qualifiers, it's not a pointer
  rc_return false;
}

bool QualifiedType::isPointer() const {
  return isPointerImpl(*this);
}

static RecursiveCoroutine<bool> isConstImpl(const model::QualifiedType &QT) {
  auto *TD = dyn_cast<model::TypedefType>(QT.UnqualifiedType().get());
  if (not QT.Qualifiers().empty()) {
    // If there are qualifiers, just look at the first
    rc_return Qualifier::isConst(QT.Qualifiers().front());
  } else if (TD != nullptr) {
    // If there are no qualifiers, but it's a typedef, traverse it
    rc_return rc_recur isConstImpl(TD->UnderlyingType());
  }

  // If there are no qualifiers, and it's not a typedef, it's not const.
  rc_return false;
}

bool QualifiedType::isConst() const {
  return isConstImpl(*this);
}

static RecursiveCoroutine<bool>
isPrimitiveImpl(const model::QualifiedType &QT,
                std::optional<model::PrimitiveTypeKind::Values> V) {
  if (QT.Qualifiers().size() != 0
      and not llvm::all_of(QT.Qualifiers(), Qualifier::isConst))
    rc_return false;

  const model::Type *UnqualifiedType = QT.UnqualifiedType().get();
  if (auto *Primitive = llvm::dyn_cast<PrimitiveType>(UnqualifiedType))
    rc_return !V.has_value() || Primitive->PrimitiveKind() == *V;

  if (auto *Typedef = llvm::dyn_cast<TypedefType>(UnqualifiedType))
    rc_return rc_recur isPrimitiveImpl(Typedef->UnderlyingType(), V);

  rc_return false;
}

bool QualifiedType::isPrimitive() const {
  return isPrimitiveImpl(*this, std::nullopt);
}

bool QualifiedType::isPrimitive(PrimitiveTypeKind::Values V) const {
  return isPrimitiveImpl(*this, V);
}

static RecursiveCoroutine<bool> isImpl(const model::QualifiedType &QT,
                                       model::TypeKind::Values K) {
  if (QT.Qualifiers().size() != 0
      and not llvm::all_of(QT.Qualifiers(), Qualifier::isConst))
    rc_return false;

  const model::Type *UnqualifiedType = QT.UnqualifiedType().get();

  if (UnqualifiedType->Kind() == K)
    rc_return true;

  if (auto *Typedef = llvm::dyn_cast<TypedefType>(UnqualifiedType))
    rc_return rc_recur isImpl(Typedef->UnderlyingType(), K);

  rc_return false;
}

bool QualifiedType::is(model::TypeKind::Values K) const {
  return isImpl(*this, K);
}

std::optional<uint64_t> Type::size() const {
  VerifyHelper VH;
  return size(VH);
}

std::optional<uint64_t> Type::trySize() const {
  VerifyHelper VH;
  return trySize(VH);
}

std::optional<uint64_t> Type::size(VerifyHelper &VH) const {
  std::optional<uint64_t> MaybeSize = trySize(VH);
  revng_check(MaybeSize);
  if (*MaybeSize == 0)
    return std::nullopt;
  else
    return MaybeSize;
}

// NOTE: there's a really similar function for computing alignment in
//       `lib/ABI/Definition.cpp`. It's better if two are kept in sync, so
//       when modifying this function, please apply corresponding modifications
//       to its little brother as well.
RecursiveCoroutine<std::optional<uint64_t>>
Type::trySize(VerifyHelper &VH) const {
  auto MaybeSize = VH.size(this);
  if (MaybeSize)
    rc_return MaybeSize;

  // This code assumes that the type T is well formed.
  uint64_t Size = 0;

  switch (Kind()) {
  case TypeKind::RawFunctionType:
  case TypeKind::CABIFunctionType:
    // Function prototypes have no size
    rc_return std::nullopt;

  case TypeKind::PrimitiveType: {
    auto *P = cast<PrimitiveType>(this);

    if (P->PrimitiveKind() == model::PrimitiveTypeKind::Void) {
      // Void types have no size
      revng_assert(P->Size() == 0);

      Size = 0;
    } else {
      Size = P->Size();
    }
  } break;

  case TypeKind::EnumType: {
    auto *U = llvm::cast<EnumType>(this);
    auto MaybeSize = rc_recur U->UnderlyingType().trySize(VH);
    if (not MaybeSize)
      rc_return std::nullopt;

    Size = *MaybeSize;
  } break;

  case TypeKind::TypedefType: {
    auto *Typedef = llvm::cast<TypedefType>(this);

    auto MaybeSize = rc_recur Typedef->UnderlyingType().trySize(VH);
    if (not MaybeSize)
      rc_return std::nullopt;

    Size = *MaybeSize;
  } break;

  case TypeKind::StructType: {
    Size = llvm::cast<StructType>(this)->Size();
  } break;

  case TypeKind::UnionType: {
    auto *U = llvm::cast<UnionType>(this);
    uint64_t Max = 0ULL;

    for (const auto &Field : U->Fields()) {
      auto MaybeFieldSize = rc_recur Field.Type().trySize(VH);
      if (not MaybeFieldSize)
        rc_return std::nullopt;

      Max = std::max(Max, *MaybeFieldSize);
    }

    Size = Max;
  } break;

  case TypeKind::Invalid:
  case TypeKind::Count:
  default:
    revng_abort();
  }

  VH.setSize(this, Size);

  rc_return Size;
};

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const PrimitiveType *T) {
  revng_assert(T->Kind() == TypeKind::PrimitiveType);

  if (not T->CustomName().empty() or not T->OriginalName().empty())
    rc_return VH.fail("PrimitiveTypes cannot have OriginalName or CustomName",
                      *T);

  auto ExpectedID = makePrimitiveID(T->PrimitiveKind(), T->Size());
  if (T->ID() != ExpectedID)
    rc_return VH.fail(Twine("Wrong ID for PrimitiveType. Got: ")
                        + Twine(T->ID()) + ". Expected: " + Twine(ExpectedID)
                        + ".",
                      *T);

  if (not isValidPrimitiveSize(T->PrimitiveKind(), T->Size()))
    rc_return VH.fail("Invalid PrimitiveType size: " + Twine(T->Size()), *T);

  rc_return true;
}

bool Identifier::verify() const {
  return verify(false);
}

bool Identifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

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
                      Twine(*this) + " is not a valid identifier");
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const EnumType *T) {
  if (T->Kind() != TypeKind::EnumType or T->Entries().empty()
      or not T->CustomName().verify(VH))
    rc_return VH.fail();

  // The underlying type has to be an unqualified primitive type
  if (not rc_recur T->UnderlyingType().verify(VH)
      or not T->UnderlyingType().Qualifiers().empty())
    rc_return VH.fail();

  // We only allow signed/unsigned as underlying type
  if (not T->UnderlyingType().isPrimitive(PrimitiveTypeKind::Signed)
      and not T->UnderlyingType().isPrimitive(PrimitiveTypeKind::Unsigned))
    rc_return VH.fail("UnderlyingType of a EnumType can only be Signed or "
                      "Unsigned",
                      *T);

  for (auto &Entry : T->Entries()) {

    if (not Entry.verify(VH))
      rc_return VH.fail();

    // TODO: verify Entry.Value is within boundaries
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const TypedefType *T) {
  rc_return VH.maybeFail(T->CustomName().verify(VH)
                         and T->Kind() == TypeKind::TypedefType
                         and rc_recur T->UnderlyingType().verify(VH));
}

inline RecursiveCoroutine<bool> isScalarImpl(const QualifiedType &QT) {
  for (const Qualifier &Q : QT.Qualifiers()) {
    switch (Q.Kind()) {
    case QualifierKind::Invalid:
      revng_abort();
    case QualifierKind::Pointer:
      rc_return true;
    case QualifierKind::Array:
      rc_return false;
    case QualifierKind::Const:
      break;
    default:
      revng_abort();
    }
  }

  const Type *Unqualified = QT.UnqualifiedType().get();
  revng_assert(Unqualified != nullptr);
  if (llvm::isa<model::PrimitiveType>(Unqualified)
      or llvm::isa<model::EnumType>(Unqualified)) {
    rc_return true;
  }

  if (auto *Typedef = llvm::dyn_cast<model::TypedefType>(Unqualified))
    rc_return rc_recur isScalarImpl(Typedef->UnderlyingType());

  rc_return false;
}

bool model::QualifiedType::isScalar() const {
  return isScalarImpl(*this);
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const StructType *T) {
  using namespace llvm;

  revng_assert(T->Kind() == TypeKind::StructType);

  if (not T->CustomName().verify(VH))
    rc_return VH.fail("Invalid name", *T);

  if (T->Size() == 0)
    rc_return VH.fail("Struct type has zero size", *T);

  size_t Index = 0;
  llvm::SmallSet<llvm::StringRef, 8> Names;
  auto FieldIt = T->Fields().begin();
  auto FieldEnd = T->Fields().end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail("Can't verify type of field at offset "
                          + Twine(Field.Offset()),
                        *T);

    if (Field.Offset() >= T->Size()) {
      std::uint64_t Size = *Field.Type().size();
      rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                          + " is out of struct boundaries (field size: "
                          + Twine(Size) + ", field offset + size: "
                          + Twine(Field.Offset() + Size)
                          + ", struct size: " + Twine(T->Size()) + ")",
                        *T);
    }

    auto MaybeSize = rc_recur Field.Type().size(VH);
    // This is verified AggregateField::verify
    revng_assert(MaybeSize);

    auto FieldEndOffset = Field.Offset() + *MaybeSize;
    auto NextFieldIt = std::next(FieldIt);
    if (NextFieldIt != FieldEnd) {
      // If this field is not the last, check that it does not overlap with the
      // following field.
      if (FieldEndOffset > NextFieldIt->Offset()) {
        rc_return VH.fail("Field at offset " + Twine(Field.Offset())
                            + " (with size: " + Twine(*Field.Type().size())
                            + ") overlaps with the field at offset "
                            + Twine(NextFieldIt->Offset()) + " (with size: "
                            + Twine(*NextFieldIt->Type().size()) + ")",
                          *T);
      }
    } else if (FieldEndOffset > T->Size()) {
      // Otherwise, if this field is the last, check that it's not larger than
      // size.
      rc_return VH.fail("Last field ends outside the struct", *T);
    }

    if (isVoidConst(&Field.Type()).IsVoid)
      rc_return VH.fail("Field " + Twine(Index + 1) + " is void", *T);

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName()))
        rc_return VH.fail("Field name collides with global symbol", *T);

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in struct fields names", *T);
    }

    ++Index;
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const UnionType *T) {
  revng_assert(T->Kind() == TypeKind::UnionType);

  if (not T->CustomName().verify(VH))
    rc_return VH.fail("Invalid name", *T);

  if (T->Fields().empty())
    rc_return VH.fail("Union type has zero fields", *T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T->Fields())) {
    auto &Field = Group.value();
    uint64_t ExpectedIndex = Group.index();

    if (Field.Index() != ExpectedIndex) {
      rc_return VH.fail(Twine("Union type is missing field ")
                          + Twine(ExpectedIndex),
                        *T);
    }

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail();

    auto MaybeSize = rc_recur Field.Type().size(VH);
    // This is verified AggregateField::verify
    revng_assert(MaybeSize);

    if (isVoidConst(&Field.Type()).IsVoid) {
      rc_return VH.fail("Field " + Twine(Field.Index()) + " is void", *T);
    }

    // Verify CustomName for collisions
    if (not Field.CustomName().empty()) {
      if (VH.isGlobalSymbol(Field.CustomName()))
        rc_return VH.fail("Field name collides with global symbol", *T);

      if (not Names.insert(Field.CustomName()).second)
        rc_return VH.fail("Collision in union fields names", *T);
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const CABIFunctionType *T) {
  if (not T->CustomName().verify(VH) or T->Kind() != TypeKind::CABIFunctionType
      or not rc_recur T->ReturnType().verify(VH))
    rc_return VH.fail();

  if (T->ABI() == model::ABI::Invalid)
    rc_return VH.fail("An invalid ABI", *T);

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T->Arguments())) {
    auto &Argument = Group.value();
    uint64_t ArgPos = Group.index();

    if (not Argument.CustomName().verify(VH))
      rc_return VH.fail("An argument has invalid CustomName", *T);

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", *T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", *T);
    }

    if (Argument.Index() != ArgPos)
      rc_return VH.fail("An argument has invalid index", *T);

    if (not rc_recur Argument.Type().verify(VH))
      rc_return VH.fail("An argument has invalid type", *T);

    VoidConstResult VoidConst = isVoidConst(&Argument.Type());
    if (VoidConst.IsVoid) {
      // If we have a void argument it must be the only one, and the function
      // cannot be vararg.
      if (T->Arguments().size() > 1)
        rc_return VH.fail("More than 1 void argument", *T);

      // Cannot have const-qualified void as argument.
      if (VoidConst.IsConst)
        rc_return VH.fail("Cannot have const void argument", *T);
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool> verifyImpl(VerifyHelper &VH,
                                           const RawFunctionType *T) {

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (const NamedTypedRegister &Argument : T->Arguments()) {
    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();

    // Verify CustomName for collisions
    if (not Argument.CustomName().empty()) {
      if (VH.isGlobalSymbol(Argument.CustomName()))
        rc_return VH.fail("Argument name collides with global symbol", *T);

      if (not Names.insert(Argument.CustomName()).second)
        rc_return VH.fail("Collision in argument names", *T);
    }
  }

  for (const TypedRegister &Return : T->ReturnValues())
    if (not rc_recur Return.verify(VH))
      rc_return VH.fail();

  for (const Register::Values &Preserved : T->PreservedRegisters())
    if (Preserved == Register::Invalid)
      rc_return VH.fail();

  if (not T->StackArgumentsType().Qualifiers().empty())
    rc_return VH.fail();
  if (auto &Type = T->StackArgumentsType().UnqualifiedType();
      Type.isValid() and not rc_recur Type.get()->verify(VH))
    rc_return VH.fail();

  rc_return VH.maybeFail(T->CustomName().verify(VH));
}

void Type::dump() const {
  auto *This = this;
  auto Dump = [](auto &Upcasted) { serialize(dbg, Upcasted); };
  upcast(This, Dump);
}

void Type::dumpTypeGraph(const char *Path) const {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

bool Type::verify() const {
  return verify(false);
}

bool Type::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> Type::verify(VerifyHelper &VH) const {
  if (VH.isVerified(this))
    rc_return true;

  // Ensure we have not infinite recursion
  if (VH.isVerificationInProgress(this))
    rc_return VH.fail();

  VH.verificationInProgress(this);

  if (ID() == 0)
    rc_return VH.fail("A type cannot have ID 0", *this);

  bool Result = false;

  // We could use upcast() but we'd need to workaround coroutines.
  switch (Kind()) {
  case TypeKind::PrimitiveType:
    Result = rc_recur verifyImpl(VH, cast<PrimitiveType>(this));
    break;

  case TypeKind::EnumType:
    Result = rc_recur verifyImpl(VH, cast<EnumType>(this));
    break;

  case TypeKind::TypedefType:
    Result = rc_recur verifyImpl(VH, cast<TypedefType>(this));
    break;

  case TypeKind::StructType:
    Result = rc_recur verifyImpl(VH, cast<StructType>(this));
    break;

  case TypeKind::UnionType:
    Result = rc_recur verifyImpl(VH, cast<UnionType>(this));
    break;

  case TypeKind::CABIFunctionType:
    Result = rc_recur verifyImpl(VH, cast<CABIFunctionType>(this));
    break;

  case TypeKind::RawFunctionType:
    Result = rc_recur verifyImpl(VH, cast<RawFunctionType>(this));
    break;

  default: // Do nothing;
    ;
  }

  if (Result) {
    VH.setVerified(this);
    VH.verificationCompleted(this);
  }

  rc_return VH.maybeFail(Result);
}

void QualifiedType::dump() const {
  serialize(dbg, *this);
}

bool QualifiedType::verify() const {
  return verify(false);
}

bool QualifiedType::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> QualifiedType::verify(VerifyHelper &VH) const {
  if (not UnqualifiedType().isValid())
    rc_return VH.fail("Underlying type is invalid", *this);

  // Verify the qualifiers are valid
  for (const auto &Q : Qualifiers())
    if (not Q.verify(VH))
      rc_return VH.fail("Invalid qualifier", Q);

  auto QIt = Qualifiers().begin();
  auto QEnd = Qualifiers().end();
  for (; QIt != QEnd; ++QIt) {
    const auto &Q = *QIt;
    auto NextQIt = std::next(QIt);
    bool HasNext = NextQIt != QEnd;

    // Check that we have not two consecutive const qualifiers
    if (HasNext and Qualifier::isConst(Q) and Qualifier::isConst(*NextQIt))
      rc_return VH.fail("QualifiedType has two consecutive const qualifiers",
                        *this);

    if (Qualifier::isPointer(Q)) {
      // Don't proceed the verification, just make sure the pointer is either
      // 32- or 64-bit
      rc_return VH.maybeFail(Q.Size() == 4 or Q.Size() == 8,
                             "Only 32-bit and 64-bit pointers "
                             "are currently "
                             "supported",
                             *this);

    } else if (Qualifier::isArray(Q)) {
      // Ensure there's at least one element
      if (Q.Size() < 1)
        rc_return VH.fail("Arrays need to have at least an element", *this);

      // Verify element type
      QualifiedType ElementType{ UnqualifiedType(), { NextQIt, QEnd } };
      if (not rc_recur ElementType.verify(VH))
        rc_return VH.fail("Array element invalid", ElementType);

      // Ensure the element type has a size and stop
      auto MaybeSize = rc_recur ElementType.size(VH);
      rc_return VH.maybeFail(MaybeSize.has_value(),
                             "Cannot compute array size",
                             ElementType);
    } else if (Qualifier::isConst(Q)) {
      // const qualifiers must have zero size
      if (Q.Size() != 0)
        rc_return VH.fail("const qualifier has non-0 size");

    } else {
      revng_abort();
    }
  }

  // If we get here, we either have no qualifiers or just const qualifiers:
  // recur on the underlying type
  rc_return VH.maybeFail(rc_recur UnqualifiedType().get()->verify(VH));
}

template<typename T>
RecursiveCoroutine<bool>
verifyTypedRegisterCommon(const T &TypedRegister, VerifyHelper &VH) {
  // Ensure the type we're pointing to is scalar
  if (not TypedRegister->Type().isScalar())
    rc_return VH.fail();

  if (TypedRegister->Location() == Register::Invalid)
    rc_return VH.fail();

  // Ensure if fits in the corresponding register
  auto MaybeTypeSize = rc_recur TypedRegister->Type().size(VH);

  // Zero-sized types are not allowed
  if (not MaybeTypeSize)
    rc_return VH.fail();

  size_t RegisterSize = model::Register::getSize(TypedRegister->Location());
  if (*MaybeTypeSize > RegisterSize)
    rc_return VH.fail();

  rc_return VH.maybeFail(rc_recur TypedRegister->Type().verify(VH));
}

void TypedRegister::dump() const {
  serialize(dbg, *this);
}

bool TypedRegister::verify() const {
  return verify(false);
}

bool TypedRegister::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> TypedRegister::verify(VerifyHelper &VH) const {
  rc_return verifyTypedRegisterCommon(this, VH);
}

void NamedTypedRegister::dump() const {
  serialize(dbg, *this);
}

bool NamedTypedRegister::verify() const {
  return verify(false);
}

bool NamedTypedRegister::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> NamedTypedRegister::verify(VerifyHelper &VH) const {
  // Ensure the name is valid
  if (not CustomName().verify(VH))
    rc_return VH.fail();

  rc_return verifyTypedRegisterCommon(this, VH);
}

bool StructField::verify() const {
  return verify(false);
}

bool StructField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> StructField::verify(VerifyHelper &VH) const {
  if (not rc_recur Type().verify(VH))
    rc_return VH.fail("Aggregate field type is not valid");

  // Aggregated fields cannot be zero-sized fields
  auto MaybeSize = rc_recur Type().size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Aggregate field is zero-sized");

  rc_return VH.maybeFail(CustomName().verify(VH));
}

bool UnionField::verify() const {
  return verify(false);
}

bool UnionField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> UnionField::verify(VerifyHelper &VH) const {
  if (not rc_recur Type().verify(VH))
    rc_return VH.fail("Aggregate field type is not valid");

  // Aggregated fields cannot be zero-sized fields
  auto MaybeSize = rc_recur Type().size(VH);
  if (not MaybeSize)
    rc_return VH.fail("Aggregate field is zero-sized", Type());

  rc_return VH.maybeFail(CustomName().verify(VH));
}

void Argument::dump() const {
  serialize(dbg, *this);
}

bool Argument::verify() const {
  return verify(false);
}

bool Argument::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> Argument::verify(VerifyHelper &VH) const {
  rc_return VH.maybeFail(CustomName().verify(VH)
                         and rc_recur Type().verify(VH));
}

} // namespace model

template model::TypePath
model::TypePath::fromString<model::Binary>(model::Binary *Root,
                                           llvm::StringRef Path);

template model::TypePath
model::TypePath::fromString<const model::Binary>(const model::Binary *Root,
                                                 llvm::StringRef Path);
