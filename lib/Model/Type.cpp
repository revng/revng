//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <bit>
#include <cstddef>
#include <random>
#include <string>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

#include "revng/Model/ABI.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

using llvm::cast;
using llvm::dyn_cast;

namespace model {

const Identifier Identifier::Empty = Identifier("");

const std::set<llvm::StringRef> ReservedKeywords = {
  // reserved keywords for primitive types
  "void"
  "pointer_or_number8_t"
  "pointer_or_number16_t"
  "pointer_or_number32_t"
  "pointer_or_number64_t"
  "pointer_or_number128_t"
  "number8_t"
  "number16_t"
  "number32_t"
  "number64_t"
  "number128_t"
  "generic8_t"
  "generic16_t"
  "generic32_t"
  "generic64_t"
  "generic128_t"
  "int8_t"
  "int16_t"
  "int32_t"
  "int64_t"
  "int128_t"
  "uint8_t"
  "uint16_t"
  "uint32_t"
  "uint64_t"
  "uint128_t"
  "float16_t"
  "float32_t"
  "float64_t"
  "float128_t"
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

static llvm::cl::opt<uint64_t> ModelTypeIDSeed("model-type-id-seed",
                                               llvm::cl::desc("Set the seed "
                                                              "for the "
                                                              "generation of "
                                                              "ID of model "
                                                              "Types"),
                                               llvm::cl::cat(MainCategory),
                                               llvm::cl::init(false));

class RNG {
  std::mt19937_64 Generator;
  std::uniform_int_distribution<uint64_t> Distribution;

public:
  RNG() :
    Generator(ModelTypeIDSeed.getNumOccurrences() ? ModelTypeIDSeed.getValue() :
                                                    std::random_device()()),
    Distribution(std::numeric_limits<uint64_t>::min(),
                 std::numeric_limits<uint64_t>::max()) {}

  uint64_t get() { return Distribution(Generator); }
};

static llvm::ManagedStatic<RNG> IDGenerator;

model::Type::Type(TypeKind::Values TK) :
  model::Type::Type(TK, IDGenerator->get()) {
}

Identifier model::UnionField::name() const {
  using llvm::Twine;
  Identifier Result;
  if (CustomName.empty())
    (Twine("unnamed_field_") + Twine(Index)).toVector(Result);
  else
    Result = CustomName;
  return Result;
}

Identifier model::StructField::name() const {
  using llvm::Twine;
  Identifier Result;
  if (CustomName.empty())
    (Twine("unnamed_field_at_offset_") + Twine(Offset)).toVector(Result);
  else
    Result = CustomName;
  return Result;
}

Identifier model::Argument::name() const {
  using llvm::Twine;
  Identifier Result;
  if (CustomName.empty())
    (Twine("unnamed_arg_") + Twine(Index)).toVector(Result);
  else
    Result = CustomName;
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
  switch (Kind) {
  case QualifierKind::Invalid:
    return VH.fail();
  case QualifierKind::Pointer:
    return VH.maybeFail(Size > 0 and llvm::isPowerOf2_64(Size));
  case QualifierKind::Const:
    return VH.maybeFail(Size == 0);
  case QualifierKind::Array:
    return VH.maybeFail(Size > 0);
  default:
    revng_abort();
  }

  return VH.fail();
}

static constexpr bool
isValidPrimitiveSize(PrimitiveTypeKind::Values PrimKind, uint8_t BS) {
  switch (PrimKind) {
  case PrimitiveTypeKind::Invalid:
    return false;

  case PrimitiveTypeKind::Void:
    return BS == 0;

  case PrimitiveTypeKind::Generic:
  case PrimitiveTypeKind::PointerOrNumber:
  case PrimitiveTypeKind::Number:
  case PrimitiveTypeKind::Unsigned:
  case PrimitiveTypeKind::Signed:
    return BS == 1 or BS == 2 or BS == 4 or BS == 8 or BS == 16;

  case PrimitiveTypeKind::Float:
    return BS == 2 or BS == 4 or BS == 8 or BS == 12 or BS == 16;

  default:
    revng_abort();
  }

  revng_abort();
}

Identifier model::PrimitiveType::name() const {
  using llvm::Twine;
  Identifier Result;

  switch (PrimitiveKind) {
  case PrimitiveTypeKind::Void:
    Result = "void";
    break;

  case PrimitiveTypeKind::Unsigned:
    (Twine("uint") + Twine(Size * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Number:
    (Twine("number") + Twine(Size * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::PointerOrNumber:
    ("pointer_or_number" + Twine(Size * 8) + "_t").toVector(Result);
    break;

  case PrimitiveTypeKind::Generic:
    (Twine("generic") + Twine(Size * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Signed:
    (Twine("int") + Twine(Size * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveTypeKind::Float:
    (Twine("float") + Twine(Size * 8) + Twine("_t")).toVector(Result);
    break;

  default:
    revng_abort();
  }

  return Result;
}

template<typename T>
Identifier customNameOrAutomatic(T *This) {
  using llvm::Twine;
  if (not This->CustomName.empty())
    return This->CustomName;
  else
    return Identifier((Twine(T::AutomaticNamePrefix) + Twine(This->ID)).str());
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

Identifier model::UnionType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::NamedTypedRegister::name() const {
  using llvm::Twine;
  if (not CustomName.empty())
    return CustomName;
  else
    return Identifier(model::Register::getRegisterName(Location));
}

Identifier model::RawFunctionType::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::CABIFunctionType::name() const {
  return customNameOrAutomatic(this);
}

static uint64_t
makePrimitiveID(PrimitiveTypeKind::Values PrimitiveKind, uint8_t Size) {
  return (static_cast<uint8_t>(PrimitiveKind) << 8) | Size;
}

static PrimitiveTypeKind::Values getPrimitiveKind(uint64_t ID) {
  return static_cast<PrimitiveTypeKind::Values>(ID >> 8);
}

static uint8_t getPrimitiveSize(uint64_t ID) {
  return ID & ((1 << 8) - 1);
}

PrimitiveType::PrimitiveType(PrimitiveTypeKind::Values PrimitiveKind,
                             uint8_t Size) :
  Type(AssociatedKind, makePrimitiveID(PrimitiveKind, Size)),
  PrimitiveKind(PrimitiveKind),
  Size(Size) {
}

PrimitiveType::PrimitiveType(uint64_t ID) :
  Type(AssociatedKind, ID),
  PrimitiveKind(getPrimitiveKind(ID)),
  Size(getPrimitiveSize(ID)) {
}

static bool beginsWithReservedPrefix(llvm::StringRef Name) {
  return Name.startswith("unnamed_");
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
  for (const Identifier &Alias : Aliases)
    if (not Alias.verify(VH))
      return VH.fail();

  return VH.maybeFail(CustomName.verify(VH) and not Aliases.count(CustomName)
                      and not Aliases.count(Identifier::Empty));
}

static bool isOnlyConstQualified(const QualifiedType &QT) {
  if (QT.Qualifiers.empty() or QT.Qualifiers.size() > 1)
    return false;

  return QT.Qualifiers[0].isConstQualifier();
}

struct VoidConstResult {
  bool IsVoid;
  bool IsConst;
};

static VoidConstResult isVoidConst(const QualifiedType *QualType) {
  VoidConstResult Result{ /* IsVoid */ false, /* IsConst */ false };

  bool Done = false;
  while (not Done) {

    // If the argument type is qualified try to get the unqualified version.
    // Warning: we only skip const-qualifiers here, cause the other qualifiers
    // actually produce a different type.
    const Type *UnqualType = nullptr;
    if (not QualType->Qualifiers.empty()) {

      // If it has a non-const qualifier, it can never be void because it's a
      // pointer or array, so we can break out.
      if (not isOnlyConstQualified(*QualType)) {
        Done = true;
        continue;
      }

      // We know that it's const-qualified here, and it only has one
      // qualifier, hence we can skip the const-qualifier.
      Result.IsConst = true;
      if (not QualType->UnqualifiedType.Root)
        return Result;
    }

    UnqualType = QualType->UnqualifiedType.get();

    switch (UnqualType->Kind) {

    // If we still have a typedef in our way, unwrap it and keep looking.
    case TypeKind::Typedef: {
      QualType = &cast<TypedefType>(UnqualType)->UnderlyingType;
    } break;

    // If we have a primitive type, check the name, and we're done.
    case TypeKind::Primitive: {
      auto *P = cast<PrimitiveType>(UnqualType);
      Result.IsVoid = P->PrimitiveKind == PrimitiveTypeKind::Void;
      Done = true;
    } break;

    // In all the other cases it's not void, break from the while.
    default: {
      Done = true;
    } break;
    }
  }
  return Result;
}

std::optional<uint64_t> QualifiedType::size() const {
  VerifyHelper VH;
  return size(VH);
}

RecursiveCoroutine<std::optional<uint64_t>>
QualifiedType::size(VerifyHelper &VH) const {
  // This code assumes that the QualifiedType QT is well formed.
  auto QIt = Qualifiers.begin();
  auto QEnd = Qualifiers.end();

  for (; QIt != QEnd; ++QIt) {

    auto &Q = *QIt;
    switch (Q.Kind) {

    case QualifierKind::Invalid:
      revng_abort();

    case QualifierKind::Pointer:
      // If we find a pointer, we're done
      rc_return Q.Size;

    case QualifierKind::Array: {
      // The size is equal to (number of elements of the array) * (size of a
      // single element).
      const QualifiedType ArrayElem{ UnqualifiedType,
                                     { std::next(QIt), QEnd } };
      auto MaybeSize = rc_recur ArrayElem.size(VH);
      revng_assert(MaybeSize);
      rc_return *MaybeSize *Q.Size;
    }

    case QualifierKind::Const:
      // Do nothing, just skip over it
      break;

    default:
      revng_abort();
    }
  }

  rc_return rc_recur UnqualifiedType.get()->size(VH);
}

inline RecursiveCoroutine<bool>
isPrimitive(const model::QualifiedType &QT,
            model::PrimitiveTypeKind::Values V) {
  auto IsConstQualifier = [](const Qualifier &Q) {
    return Q.Kind == model::QualifierKind::Const;
  };

  if (QT.Qualifiers.size() != 0
      and not llvm::all_of(QT.Qualifiers, IsConstQualifier))
    rc_return false;

  const model::Type *UnqualifiedType = QT.UnqualifiedType.get();
  if (auto *Primitive = llvm::dyn_cast<PrimitiveType>(UnqualifiedType))
    rc_return Primitive->PrimitiveKind == V;
  else if (auto *Typedef = llvm::dyn_cast<TypedefType>(UnqualifiedType))
    rc_return rc_recur isPrimitive(Typedef->UnderlyingType, V);

  rc_return false;
}

bool QualifiedType::isPrimitive(model::PrimitiveTypeKind::Values V) const {
  return model::isPrimitive(*this, V);
}

std::optional<uint64_t> Type::size() const {
  VerifyHelper VH;
  return size(VH);
}

RecursiveCoroutine<std::optional<uint64_t>> Type::size(VerifyHelper &VH) const {
  using ResultType = std::optional<uint64_t>;
  auto MaybeSize = VH.size(this);
  if (MaybeSize)
    rc_return{ *MaybeSize == 0 ? ResultType{} : *MaybeSize };

  // This code assumes that the type T is well formed.
  ResultType Size;

  switch (Kind) {
  case TypeKind::Invalid:
    revng_abort();

  case TypeKind::RawFunctionType:
  case TypeKind::CABIFunctionType:
    // Function prototypes have no size
    Size = {};
    break;

  case TypeKind::Primitive: {
    auto *P = cast<PrimitiveType>(this);

    if (P->PrimitiveKind == model::PrimitiveTypeKind::Void) {
      // Void types have no size
      revng_assert(P->Size == 0);
      Size = {};
    } else {
      Size = P->Size;
    }
  } break;

  case TypeKind::Enum: {
    auto *U = llvm::cast<EnumType>(this)->UnderlyingType.get();
    Size = rc_recur U->size(VH);
  } break;

  case TypeKind::Typedef: {
    auto *Typedef = llvm::cast<TypedefType>(this);
    Size = rc_recur Typedef->UnderlyingType.size(VH);
  } break;

  case TypeKind::Struct: {
    Size = llvm::cast<StructType>(this)->Size;
  } break;

  case TypeKind::Union: {
    auto *U = llvm::cast<UnionType>(this);
    uint64_t Max = 0ULL;
    for (const auto &Field : U->Fields) {
      auto FieldSize = rc_recur Field.Type.size(VH);
      Max = std::max(Max, FieldSize ? *FieldSize : 0);
    }
    Size = { Max == 0 ? ResultType{} : Max };
  } break;

  default:
    revng_abort();
  }

  VH.setSize(this, Size ? *Size : 0);

  rc_return Size;
};

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const PrimitiveType *T) {
  rc_return VH.maybeFail(T->Kind == TypeKind::Primitive
                         and makePrimitiveID(T->PrimitiveKind, T->Size) == T->ID
                         and isValidPrimitiveSize(T->PrimitiveKind, T->Size));
}

bool Identifier::verify() const {
  return verify(false);
}

bool Identifier::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

bool Identifier::verify(VerifyHelper &VH) const {
  const auto AllAlphaNumOrUnderscore = [](const auto &Range) {
    const auto IsNotUnderscore = [](const char C) { return C != '_'; };
    return llvm::all_of(llvm::make_filter_range(Range, IsNotUnderscore),
                        isalnum);
  };
  return VH.maybeFail(not(not empty() and std::isdigit((*this)[0]))
                        and not startswith("_")
                        and AllAlphaNumOrUnderscore(*this)
                        and not beginsWithReservedPrefix(*this)
                        and not ReservedKeywords.count(llvm::StringRef(*this)),
                      llvm::Twine(*this) + " is not a valid identifier");
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const EnumType *T) {
  if (T->Kind != TypeKind::Enum or T->Entries.empty()
      or not T->CustomName.verify(VH))
    rc_return VH.fail();

  // The underlying type has to be a primitive type
  if (not T->UnderlyingType.isValid())
    rc_return VH.fail();

  auto *Underlying = dyn_cast<PrimitiveType>(T->UnderlyingType.get());
  if (Underlying == nullptr)
    rc_return VH.fail();

  if (not rc_recur Underlying->verify(VH))
    rc_return VH.fail();

  // We only allow signed/unsigned as underlying type
  if (Underlying->PrimitiveKind != PrimitiveTypeKind::Signed
      and Underlying->PrimitiveKind != PrimitiveTypeKind::Unsigned)
    rc_return VH.fail();

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Entry : T->Entries) {

    if (not Entry.verify(VH))
      rc_return VH.fail();

    // TODO: verify Entry.Value is within boundaries

    if (not Entry.CustomName.empty()) {
      if (not Names.insert(Entry.CustomName).second)
        rc_return VH.fail();
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const TypedefType *T) {
  rc_return VH.maybeFail(T->CustomName.verify(VH)
                         and T->Kind == TypeKind::Typedef
                         and rc_recur T->UnderlyingType.verify(VH));
}

inline RecursiveCoroutine<bool> isScalar(const QualifiedType &QT) {
  for (const Qualifier &Q : QT.Qualifiers) {
    switch (Q.Kind) {
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

  const Type *Unqualified = QT.UnqualifiedType.get();
  revng_assert(Unqualified != nullptr);
  if (llvm::isa<model::PrimitiveType>(Unqualified)) {
    rc_return true;
  } else if (auto *Typedef = llvm::dyn_cast<model::TypedefType>(Unqualified)) {
    rc_return rc_recur isScalar(Typedef->UnderlyingType);
  }

  rc_return false;
}

bool model::QualifiedType::isScalar() const {
  return ::model::isScalar(*this);
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const StructType *T) {
  using namespace llvm;

  revng_assert(T->Kind == TypeKind::Struct);

  if (not T->CustomName.verify(VH))
    rc_return VH.fail("Invalid name", *T);

  if (T->Size == 0)
    rc_return VH.fail("Struct type has zero size", *T);

  if (T->Fields.empty())
    rc_return VH.fail("Struct has no fields", *T);

  size_t Index = 0;
  llvm::SmallSet<llvm::StringRef, 8> Names;
  auto FieldIt = T->Fields.begin();
  auto FieldEnd = T->Fields.end();
  for (; FieldIt != FieldEnd; ++FieldIt) {
    auto &Field = *FieldIt;

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail("Can't verify type of field " + llvm::Twine(Index + 1),
                        *T);

    if (Field.Offset >= T->Size)
      rc_return VH.fail("Field " + Twine(Index + 1)
                          + " out of struct boundaries (offset: "
                          + Twine(Field.Offset) + ", size: " + Twine(T->Size)
                          + ")",
                        *T);

    auto MaybeSize = rc_recur Field.Type.size(VH);

    // Structs cannot have zero-sized fields
    if (not MaybeSize)
      rc_return VH.fail("Field" + Twine(Index + 1) + " is zero-sized", *T);

    auto FieldEndOffset = Field.Offset + *MaybeSize;
    auto NextFieldIt = std::next(FieldIt);
    if (NextFieldIt != FieldEnd) {
      // If this field is not the last, check that it does not overlap with the
      // following field.
      if (FieldEndOffset > NextFieldIt->Offset)
        rc_return VH.fail("Field " + Twine(Index + 1)
                            + " overlaps with the next one",
                          *T);
    } else if (FieldEndOffset > T->Size) {
      // Otherwise, if this field is the last, check that it's not larger than
      // size.
      rc_return VH.fail("Last field ends outside the struct", *T);
    }

    if (isVoidConst(&Field.Type).IsVoid)
      rc_return VH.fail("Field " + Twine(Index + 1) + " is void", *T);

    if (not Field.CustomName.empty()
        and not Names.insert(Field.CustomName).second)
      rc_return VH.fail("Collision in struct fields names", *T);

    ++Index;
  }

  rc_return true;
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const UnionType *T) {
  if (not T->CustomName.verify(VH) or T->Kind != TypeKind::Union
      or T->Fields.empty())
    rc_return false;

  llvm::SmallSet<llvm::StringRef, 8> Names;
  for (auto &Group : llvm::enumerate(T->Fields)) {
    auto &Field = Group.value();
    uint64_t ExpectedIndex = Group.index();

    if (Field.Index != ExpectedIndex)
      rc_return VH.fail();

    if (not rc_recur Field.verify(VH))
      rc_return VH.fail();

    if (Field.CustomName.size() > 0) {
      if (not Names.insert(Field.CustomName).second)
        rc_return VH.fail();
    }

    if (isVoidConst(&Field.Type).IsVoid)
      rc_return VH.fail();
  }

  rc_return true;
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const CABIFunctionType *T) {
  if (not T->CustomName.verify(VH) or T->Kind != TypeKind::CABIFunctionType
      or not rc_recur T->ReturnType.verify(VH))
    rc_return VH.fail();

  if (T->ABI == model::abi::Invalid)
    rc_return VH.fail();

  for (auto &Group : llvm::enumerate(T->Arguments)) {
    auto &Argument = Group.value();
    uint64_t ArgPos = Group.index();

    if (not Argument.CustomName.verify(VH))
      rc_return VH.fail();

    if (Argument.Index != ArgPos)
      rc_return VH.fail();

    if (not rc_recur Argument.Type.verify(VH))
      rc_return VH.fail();

    VoidConstResult VoidConst = isVoidConst(&Argument.Type);
    if (VoidConst.IsVoid) {
      // If we have a void argument it must be the only one, and the function
      // cannot be vararg.
      if (T->Arguments.size() > 1)
        rc_return VH.fail();

      // Cannot have const-qualified void as argument.
      if (VoidConst.IsConst)
        rc_return VH.fail();
    }
  }

  rc_return true;
}

static RecursiveCoroutine<bool>
verifyImpl(VerifyHelper &VH, const RawFunctionType *T) {

  for (const NamedTypedRegister &Argument : T->Arguments)
    if (not rc_recur Argument.verify(VH))
      rc_return VH.fail();

  for (const TypedRegister &Return : T->ReturnValues)
    if (not rc_recur Return.verify(VH))
      rc_return VH.fail();

  for (const Register::Values &Preserved : T->PreservedRegisters)
    if (Preserved == Register::Invalid)
      rc_return VH.fail();

  rc_return VH.maybeFail(T->CustomName.verify(VH));
}

void Type::dump() const {
  auto *This = this;
  auto Dump = [](auto &Upcasted) { serialize(dbg, Upcasted); };
  upcast(This, Dump);
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
  if (VH.isVerificationInProgess(this))
    rc_return VH.fail();

  VH.verificationInProgess(this);

  if (ID == 0)
    rc_return VH.fail();

  bool Result = false;

  // We could use upcast() but we'd need to workaround coroutines.
  switch (Kind) {
  case TypeKind::Primitive:
    Result = rc_recur verifyImpl(VH, cast<PrimitiveType>(this));
    break;

  case TypeKind::Enum:
    Result = rc_recur verifyImpl(VH, cast<EnumType>(this));
    break;

  case TypeKind::Typedef:
    Result = rc_recur verifyImpl(VH, cast<TypedefType>(this));
    break;

  case TypeKind::Struct:
    Result = rc_recur verifyImpl(VH, cast<StructType>(this));
    break;

  case TypeKind::Union:
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

  if (Result)
    VH.setVerified(this);

  VH.verificationCompleted(this);

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
  if (not UnqualifiedType.isValid())
    rc_return VH.fail();

  // Verify the qualifiers are valid
  for (const auto &Q : Qualifiers)
    if (not Q.verify(VH))
      rc_return VH.fail();

  auto QIt = Qualifiers.begin();
  auto QEnd = Qualifiers.end();
  for (; QIt != QEnd; ++QIt) {
    const auto &Q = *QIt;
    auto NextQIt = std::next(QIt);
    bool HasNext = NextQIt != QEnd;

    // Check that we have not two consecutive const qualifiers
    if (HasNext and Q.isConstQualifier() and NextQIt->isConstQualifier())
      rc_return VH.fail();

    if (Q.isPointerQualifier()) {
      // Don't proceed the verification, just make sure the pointer is either
      // 32- or 64-bits
      rc_return VH.maybeFail(Q.Size == 4 or Q.Size == 8,
                             "Only 32-bit and 64-bit pointers are currently "
                             "supported");

    } else if (Q.isArrayQualifier()) {
      // Ensure there's at least one element
      if (Q.Size < 1)
        rc_return VH.fail("Arrays need to have at least an element");

      // Verify element type
      QualifiedType ElementType{ UnqualifiedType, { NextQIt, QEnd } };
      if (not rc_recur ElementType.verify(VH))
        rc_return VH.fail();

      // Ensure the element type has a size and stop
      auto MaybeSize = rc_recur ElementType.size(VH);
      rc_return VH.maybeFail(MaybeSize.has_value());
    } else if (Q.isConstQualifier()) {
      // const qualifiers must have zero size
      if (Q.Size != 0)
        rc_return VH.fail();

    } else {
      revng_abort();
    }
  }

  // If we get here, we either have no qualifiers or just const qualifiers:
  // recur on the underlying type
  rc_return VH.maybeFail(rc_recur UnqualifiedType.get()->verify(VH));
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
  // Ensure the type we're pointing to is scalar
  if (not isScalar(Type))
    rc_return VH.fail();

  if (Location == Register::Invalid)
    rc_return VH.fail();

  // Ensure if fits in the corresponding register
  auto MaybeTypeSize = rc_recur Type.size(VH);

  // Zero-sized types are not allowed
  if (not MaybeTypeSize)
    rc_return VH.fail();

  size_t RegisterSize = model::Register::getSize(Location);
  if (*MaybeTypeSize > RegisterSize)
    rc_return VH.fail();

  rc_return VH.maybeFail(rc_recur Type.verify(VH));
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
  const TypedRegister &TR = *this;
  rc_return VH.maybeFail(CustomName.verify(VH) and rc_recur TR.verify(VH));
}

bool AggregateField::verify() const {
  return verify(false);
}

bool AggregateField::verify(bool Assert) const {
  VerifyHelper VH(Assert);
  return verify(VH);
}

RecursiveCoroutine<bool> AggregateField::verify(VerifyHelper &VH) const {
  rc_return VH.maybeFail(CustomName.verify(VH) and rc_recur Type.verify(VH));
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
  rc_return VH.maybeFail(CustomName.verify(VH) and rc_recur Type.verify(VH));
}

} // namespace model
