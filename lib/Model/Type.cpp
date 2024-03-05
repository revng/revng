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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/VerifyHelper.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::Twine;

namespace model {

model::TypeDefinition::TypeDefinition() :
  model::TypeDefinition(0, model::TypeDefinitionKind::Invalid){};

model::TypeDefinition::TypeDefinition(uint64_t ID,
                                      TypeDefinitionKind::Values Kind) :
  model::generated::TypeDefinition(ID, Kind) {
}

const llvm::SmallVector<model::QualifiedType, 4>
model::TypeDefinition::edges() const {
  const auto *This = this;
  auto GetEdges = [](const auto &Upcasted) { return Upcasted.edges(); };
  return upcast(This, GetEdges, llvm::SmallVector<model::QualifiedType, 4>());
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

Identifier model::TypeDefinition::name() const {
  auto *This = this;
  auto GetName = [](auto &Upcasted) -> Identifier { return Upcasted.name(); };
  return upcast(This, GetName, Identifier(""));
}

std::optional<model::PrimitiveDefinition>
model::PrimitiveDefinition::fromName(llvm::StringRef Name) {
  PrimitiveKind::Values Kind = PrimitiveKind::Invalid;
  uint8_t Size = 0;

  // Handle void
  if (Name == "void") {
    Kind = PrimitiveKind::Void;
    return model::PrimitiveDefinition(Kind, Size);
  }

  // Ensure the name ends with _t
  if (not Name.consume_back("_t"))
    return std::nullopt;

  // Parse the prefix for the kind
  if (Name.consume_front("generic")) {
    Kind = PrimitiveKind::Generic;
  } else if (Name.consume_front("uint")) {
    Kind = PrimitiveKind::Unsigned;
  } else if (Name.consume_front("number")) {
    Kind = PrimitiveKind::Number;
  } else if (Name.consume_front("pointer_or_number")) {
    Kind = PrimitiveKind::PointerOrNumber;
  } else if (Name.consume_front("int")) {
    Kind = PrimitiveKind::Signed;
  } else if (Name.consume_front("float")) {
    Kind = PrimitiveKind::Float;
  } else {
    return std::nullopt;
  }

  // Consume bit size
  unsigned Bits = 0;
  if (Name.consumeInteger(10, Bits))
    return std::nullopt;

  // Ensure we consumed everything
  if (Name.size() != 0)
    return std::nullopt;

  // Ensure it's a multiple of 8
  if (Bits % 8 != 0)
    return std::nullopt;

  Size = Bits / 8;

  // Create the type
  model::PrimitiveDefinition NewType(Kind, Size);

  if (not NewType.verify())
    return std::nullopt;

  return NewType;
}

Identifier model::PrimitiveDefinition::name() const {
  Identifier Result;

  switch (PrimitiveKind()) {
  case PrimitiveKind::Void:
    Result = "void";
    break;

  case PrimitiveKind::Unsigned:
    (Twine("uint") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveKind::Number:
    (Twine("number") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveKind::PointerOrNumber:
    ("pointer_or_number" + Twine(Size() * 8) + "_t").toVector(Result);
    break;

  case PrimitiveKind::Generic:
    (Twine("generic") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveKind::Signed:
    (Twine("int") + Twine(Size() * 8) + Twine("_t")).toVector(Result);
    break;

  case PrimitiveKind::Float:
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

Identifier model::StructDefinition::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::TypedefDefinition::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::EnumDefinition::name() const {
  return customNameOrAutomatic(this);
}

Identifier
model::EnumDefinition::entryName(const model::EnumEntry &Entry) const {
  revng_assert(Entries().count(Entry.Value()) != 0);

  if (Entry.CustomName().size() > 0) {
    return Entry.CustomName();
  } else {
    return Identifier((Twine("_enum_entry_") + name().str() + "_"
                       + Twine(Entry.Value()))
                        .str());
  }
}

Identifier model::UnionDefinition::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::NamedTypedRegister::name() const {
  if (not CustomName().empty()) {
    return CustomName();
  } else {
    return Identifier((Twine("_register_") + getRegisterName(Location()))
                        .str());
  }
}

Identifier model::RawFunctionDefinition::name() const {
  return customNameOrAutomatic(this);
}

Identifier model::CABIFunctionDefinition::name() const {
  return customNameOrAutomatic(this);
}

static uint64_t makePrimitiveID(PrimitiveKind::Values PrimitiveKind,
                                uint8_t Size) {
  return (static_cast<uint8_t>(PrimitiveKind) << 8) | Size;
}

PrimitiveDefinition::PrimitiveDefinition(PrimitiveKind::Values PrimitiveKind,
                                         uint8_t Size) :
  PrimitiveDefinition(makePrimitiveID(PrimitiveKind, Size),
                      {},
                      {},
                      {},
                      PrimitiveKind,
                      Size) {
}

static PrimitiveKind::Values getPrimitiveKind(uint64_t ID) {
  return static_cast<PrimitiveKind::Values>(ID >> 8);
}

static uint8_t getPrimitiveSize(uint64_t ID) {
  return ID & ((1 << 8) - 1);
}

PrimitiveDefinition::PrimitiveDefinition(uint64_t ID) :
  PrimitiveDefinition(ID,
                      {},
                      {},
                      {},
                      getPrimitiveKind(ID),
                      getPrimitiveSize(ID)) {
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

  if (UnqualifiedType().empty())
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

  if (auto *TD = dyn_cast<model::TypedefDefinition>(QT.UnqualifiedType().get()))
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

  if (auto *TD = dyn_cast<model::TypedefDefinition>(QT.UnqualifiedType().get()))
    rc_return rc_recur isPointerImpl(TD->UnderlyingType());

  // If there are no non-const qualifiers, it's not a pointer
  rc_return false;
}

bool QualifiedType::isPointer() const {
  return isPointerImpl(*this);
}

static RecursiveCoroutine<bool> isConstImpl(const model::QualifiedType &QT) {
  auto *TD = dyn_cast<model::TypedefDefinition>(QT.UnqualifiedType().get());
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
                std::optional<model::PrimitiveKind::Values> V) {
  if (QT.Qualifiers().size() != 0
      and not llvm::all_of(QT.Qualifiers(), Qualifier::isConst))
    rc_return false;

  const model::TypeDefinition *UnqualifiedType = QT.UnqualifiedType().get();
  if (auto *Primitive = llvm::dyn_cast<PrimitiveDefinition>(UnqualifiedType))
    rc_return !V.has_value() || Primitive->PrimitiveKind() == *V;

  if (auto *Typedef = llvm::dyn_cast<TypedefDefinition>(UnqualifiedType))
    rc_return rc_recur isPrimitiveImpl(Typedef->UnderlyingType(), V);

  rc_return false;
}

bool QualifiedType::isPrimitive() const {
  return isPrimitiveImpl(*this, std::nullopt);
}

bool QualifiedType::isPrimitive(PrimitiveKind::Values V) const {
  return isPrimitiveImpl(*this, V);
}

static RecursiveCoroutine<bool> isImpl(const model::QualifiedType &QT,
                                       model::TypeDefinitionKind::Values K) {
  if (QT.Qualifiers().size() != 0
      and not llvm::all_of(QT.Qualifiers(), Qualifier::isConst))
    rc_return false;

  const model::TypeDefinition *UnqualifiedType = QT.UnqualifiedType().get();

  if (UnqualifiedType->Kind() == K)
    rc_return true;

  if (auto *Typedef = llvm::dyn_cast<TypedefDefinition>(UnqualifiedType))
    rc_return rc_recur isImpl(Typedef->UnderlyingType(), K);

  rc_return false;
}

bool QualifiedType::is(model::TypeDefinitionKind::Values K) const {
  return isImpl(*this, K);
}

static std::optional<model::QualifiedType>
unwrapTypedef(const model::QualifiedType &QT) {
  if (QT.UnqualifiedType().empty() or QT.Qualifiers().size() != 0)
    return std::nullopt;

  if (auto Typedef = llvm::dyn_cast<TypedefDefinition>(QT.UnqualifiedType()
                                                         .get())) {
    return Typedef->UnderlyingType();
  } else {
    return std::nullopt;
  }
}

model::QualifiedType QualifiedType::skipTypedefs() const {
  model::QualifiedType Result = *this;

  while (auto MaybeUnwrapped = unwrapTypedef(Result))
    Result = *MaybeUnwrapped;

  return Result;
}

std::optional<model::DefinitionReference>
model::QualifiedType::getFunctionType() const {
  model::QualifiedType Unwrapped = skipTypedefs();
  if (Unwrapped.Qualifiers().size() != 0 or Unwrapped.UnqualifiedType().empty())
    return nullopt;

  const model::TypeDefinition *Result = Unwrapped.UnqualifiedType().get();
  if (llvm::isa<RawFunctionDefinition>(Result)
      or llvm::isa<CABIFunctionDefinition>(Result))
    return Unwrapped.UnqualifiedType();
  else
    return nullopt;
}

std::optional<uint64_t> TypeDefinition::size() const {
  VerifyHelper VH;
  return size(VH);
}

std::optional<uint64_t> TypeDefinition::trySize() const {
  VerifyHelper VH;
  return trySize(VH);
}

std::optional<uint64_t> TypeDefinition::size(VerifyHelper &VH) const {
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
TypeDefinition::trySize(VerifyHelper &VH) const {
  // TODO: handle recursive types

  auto MaybeSize = VH.size(this);
  if (MaybeSize)
    rc_return MaybeSize;

  // This code assumes that the type T is well formed.
  uint64_t Size = 0;

  switch (Kind()) {
  case TypeDefinitionKind::RawFunctionDefinition:
  case TypeDefinitionKind::CABIFunctionDefinition:
    // Function prototypes have no size
    rc_return 0;

  case TypeDefinitionKind::PrimitiveDefinition: {
    auto *P = cast<PrimitiveDefinition>(this);

    if (P->PrimitiveKind() == model::PrimitiveKind::Void) {
      // Void types have no size
      revng_assert(P->Size() == 0);

      Size = 0;
    } else {
      Size = P->Size();
    }
  } break;

  case TypeDefinitionKind::EnumDefinition: {
    auto *U = llvm::cast<EnumDefinition>(this);
    auto MaybeSize = rc_recur U->UnderlyingType().trySize(VH);
    if (not MaybeSize)
      rc_return std::nullopt;

    Size = *MaybeSize;
  } break;

  case TypeDefinitionKind::TypedefDefinition: {
    auto *Typedef = llvm::cast<TypedefDefinition>(this);

    auto MaybeSize = rc_recur Typedef->UnderlyingType().trySize(VH);
    if (not MaybeSize)
      rc_return std::nullopt;

    Size = *MaybeSize;
  } break;

  case TypeDefinitionKind::StructDefinition: {
    Size = llvm::cast<StructDefinition>(this)->Size();
  } break;

  case TypeDefinitionKind::UnionDefinition: {
    auto *U = llvm::cast<UnionDefinition>(this);
    uint64_t Max = 0ULL;

    for (const auto &Field : U->Fields()) {
      auto MaybeFieldSize = rc_recur Field.Type().trySize(VH);
      if (not MaybeFieldSize)
        rc_return std::nullopt;

      Max = std::max(Max, *MaybeFieldSize);
    }

    Size = Max;
  } break;

  case TypeDefinitionKind::Invalid:
  case TypeDefinitionKind::Count:
  default:
    revng_abort();
  }

  VH.setSize(this, Size);

  rc_return Size;
};

void TypeDefinition::dumpTypeGraph(const char *Path) const {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC);
  if (EC)
    revng_abort(EC.message().c_str());

  TypeSystemPrinter TSPrinter(Out);
  TSPrinter.print(*this);
}

} // namespace model

using MB = model::Binary;

template model::DefinitionReference
model::DefinitionReference::fromString<MB>(MB *Root, llvm::StringRef Path);

template model::DefinitionReference
model::DefinitionReference::fromString<const MB>(const MB *Root,
                                                 llvm::StringRef Path);
