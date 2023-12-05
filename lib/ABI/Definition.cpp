/// \file Definition.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <span>
#include <unordered_map>

#include "revng/ABI/Definition.h"
#include "revng/ADT/Concepts.h"
#include "revng/Model/ABI.h"
#include "revng/Model/Binary.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Model/TypedRegister.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/YAMLTraits.h"

template<ranges::range RegisterContainer>
bool verifyRegisters(const RegisterContainer &Registers,
                     model::Architecture::Values Architecture) {
  for (const model::Register::Values &Register : Registers) {
    // Verify the architecture
    if (!model::Register::isUsedInArchitecture(Register, Architecture))
      return false;

    // Verify that there are no duplicates
    if (llvm::count(Registers, Register) != 1)
      return false;
  }

  return true;
}

static bool isVectorRegister(model::Register::Values Register) {
  using model::Register::primitiveKind;
  return primitiveKind(Register) == model::PrimitiveTypeKind::Float;
}

/// Helps detecting unsupported ABI trait definition with respect to
/// the way they return the return values.
///
/// This is an important piece of abi trait verification. For more information
/// see the `static_assert` that invokes it in \ref distributeArguments
///
/// \return `true` if the ABI is valid, `false` otherwise.
static bool verifyReturnValueLocation(const abi::Definition &D) {
  if (D.ReturnValueLocationRegister() == model::Register::Invalid) {
    // Skip ABIs that do not allow returning big values.
    // They do not benefit from this check.
    return true;
  }

  // Make sure the architecture of of the register is as expected.
  const auto Architecture = model::ABI::getRegisterArchitecture(D.ABI());
  const model::Register::Values RVLR = D.ReturnValueLocationRegister();
  if (!model::Register::isUsedInArchitecture(RVLR, Architecture))
    return false;

  if (isVectorRegister(D.ReturnValueLocationRegister())) {
    // Vector register used as the return value locations are not supported.
    return false;
  } else if (llvm::is_contained(D.CalleeSavedRegisters(),
                                D.ReturnValueLocationRegister())) {
    // Using callee saved register as a return value location doesn't make
    // much sense: filter those out.
    return false;
  } else {
    // The return value location register can optionally also be the first
    // GPRs, but only the first one.
    const auto &GPRs = D.GeneralPurposeArgumentRegisters();
    const auto Iterator = llvm::find(GPRs, D.ReturnValueLocationRegister());
    if (Iterator != GPRs.end() && Iterator != GPRs.begin())
      return false;
  }

  return true;
}

namespace abi {

bool Definition::verify() const {
  if (ABI() == model::ABI::Invalid)
    return false;

  const auto Architecture = model::ABI::getRegisterArchitecture(ABI());
  if (!verifyRegisters(GeneralPurposeArgumentRegisters(), Architecture))
    return false;
  if (!verifyRegisters(GeneralPurposeReturnValueRegisters(), Architecture))
    return false;
  if (!verifyRegisters(VectorArgumentRegisters(), Architecture))
    return false;
  if (!verifyRegisters(VectorReturnValueRegisters(), Architecture))
    return false;
  if (!verifyRegisters(CalleeSavedRegisters(), Architecture))
    return false;

  if (!verifyReturnValueLocation(*this))
    return false;

  if (ScalarTypes().empty())
    return false;

  return true;
}

using model::RawFunctionType;
bool Definition::isIncompatibleWith(const RawFunctionType &Function) const {
  revng_assert(verify());
  const auto Architecture = model::ABI::getRegisterArchitecture(ABI());

  SortedVector<model::Register::Values> Arguments;
  for (auto I = Arguments.batch_insert(); auto R : Function.Arguments()) {
    if (!model::Register::isUsedInArchitecture(R.Location(), Architecture))
      return true;

    I.emplace(R.Location());
  }

  SortedVector<model::Register::Values> AllowedArguments;
  {
    auto I = AllowedArguments.batch_insert();
    for (model::Register::Values R : GeneralPurposeArgumentRegisters())
      I.emplace(R);
    for (model::Register::Values R : VectorArgumentRegisters())
      I.emplace(R);
  }

  if (!std::includes(AllowedArguments.begin(),
                     AllowedArguments.end(),
                     Arguments.begin(),
                     Arguments.end())) {
    return true;
  }

  SortedVector<model::Register::Values> ReturnValues;
  for (auto I = ReturnValues.batch_insert(); auto R : Function.ReturnValues()) {
    if (!model::Register::isUsedInArchitecture(R.Location(), Architecture))
      return true;

    I.emplace(R.Location());
  }

  SortedVector<model::Register::Values> AllowedReturnValues;
  {
    auto I = AllowedReturnValues.batch_insert();
    for (model::Register::Values R : GeneralPurposeReturnValueRegisters())
      I.emplace(R);
    for (model::Register::Values R : VectorReturnValueRegisters())
      I.emplace(R);
  }

  if (!std::includes(AllowedReturnValues.begin(),
                     AllowedReturnValues.end(),
                     ReturnValues.begin(),
                     ReturnValues.end())) {
    return true;
  }

  for (model::Register::Values Register : Function.PreservedRegisters())
    if (!model::Register::isUsedInArchitecture(Register, Architecture))
      return true;

  return false;
}

static std::string translateABIName(model::ABI::Values ABI) {
  return "share/revng/abi/" + model::ABI::getName(ABI).str() + ".yml";
}

static std::unordered_map<model::ABI::Values, Definition> DefinitionCache;
const Definition &Definition::get(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);

  auto CacheIterator = DefinitionCache.find(ABI);
  if (CacheIterator != DefinitionCache.end()) {
    // This ABI was already loaded, grab it from the cache.
    return CacheIterator->second;
  }

  auto MaybePath = revng::ResourceFinder.findFile(translateABIName(ABI));
  if (!MaybePath.has_value()) {
    std::string Error = "The ABI definition is missing for: "
                        + serializeToString(ABI);
    revng_abort(Error.c_str());
  }

  auto Parsed = TupleTree<Definition>::fromFile(MaybePath.value());
  if (!Parsed) {
    std::string Error = "Unable to deserialize the definition for: "
                        + serializeToString(ABI);
    revng_abort(Error.c_str());
  }

  if (!Parsed->verify()) {
    std::string Error = "Deserialized ABI definition is not valid: "
                        + serializeToString(ABI);
    revng_abort(Error.c_str());
  }

  auto [It, Success] = DefinitionCache.try_emplace(ABI, std::move(**Parsed));
  revng_assert(Success);
  return It->second;
}

static RecursiveCoroutine<std::optional<uint64_t>>
naturalAlignment(const abi::Definition &ABI,
                 const model::Type &Type,
                 abi::Definition::AlignmentCache &Cache);
static RecursiveCoroutine<std::optional<uint64_t>>
naturalAlignment(const abi::Definition &ABI,
                 const model::QualifiedType &Type,
                 abi::Definition::AlignmentCache &Cache);

template<typename RealType>
RecursiveCoroutine<std::optional<uint64_t>>
underlyingAlignment(const abi::Definition &ABI,
                    const model::Type &Type,
                    abi::Definition::AlignmentCache &Cache) {
  const auto &Underlying = llvm::cast<RealType>(&Type)->UnderlyingType();
  rc_return rc_recur naturalAlignment(ABI, Underlying, Cache);
}

template<typename RealType>
RecursiveCoroutine<std::optional<uint64_t>>
fieldAlignment(const abi::Definition &ABI,
               const model::Type &Type,
               abi::Definition::AlignmentCache &Cache) {
  uint64_t Alignment = 0;
  for (const auto &Field : llvm::cast<RealType>(&Type)->Fields()) {
    if (auto A = rc_recur naturalAlignment(ABI, Field.Type(), Cache))
      Alignment = std::max(Alignment, *A);
    else
      rc_return std::nullopt;
  }

  rc_return Alignment;
}

static RecursiveCoroutine<std::optional<uint64_t>>
naturalAlignment(const abi::Definition &ABI,
                 const model::Type &Type,
                 abi::Definition::AlignmentCache &Cache) {
  if (auto Iterator = Cache.find(&Type); Iterator != Cache.end())
    rc_return Iterator->second;

  uint64_t Alignment = 0;

  // This code assumes that the type `Type` is well formed.
  switch (Type.Kind()) {
  case model::TypeKind::RawFunctionType:
  case model::TypeKind::CABIFunctionType:
    // Function prototypes have no size - hence no alignment.
    rc_return std::nullopt;

  case model::TypeKind::PrimitiveType: {
    // The alignment of primitives is simple to figure out based on the abi
    const auto *P = llvm::cast<model::PrimitiveType>(&Type);
    if (P->PrimitiveKind() == model::PrimitiveTypeKind::Void) {
      // `void` has no size either - hence no alignment.
      revng_assert(P->Size() == 0);

      Alignment = 0;
    } else if (P->PrimitiveKind() == model::PrimitiveTypeKind::Float) {
      auto Iterator = ABI.FloatingPointScalarTypes().find(P->Size());
      if (Iterator == ABI.FloatingPointScalarTypes().end())
        rc_return std::nullopt;
      Alignment = Iterator->alignedAt();
    } else {
      auto Iterator = ABI.ScalarTypes().find(P->Size());
      if (Iterator == ABI.ScalarTypes().end())
        rc_return std::nullopt;
      Alignment = Iterator->alignedAt();
    }
  } break;

  case model::TypeKind::EnumType:
    // The alignment of an enum is the same as the alignment of its underlying
    // type
    using modelEnumType = model::EnumType;
    if (auto A = rc_recur underlyingAlignment<modelEnumType>(ABI, Type, Cache))
      Alignment = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeKind::TypedefType:
    // The alignment of an enum is the same as the alignment of its underlying
    // type
    using model::TypedefType;
    if (auto A = rc_recur underlyingAlignment<TypedefType>(ABI, Type, Cache))
      Alignment = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeKind::StructType:
    // The alignment of a struct is the same as the alignment of its most
    // strictly aligned member.
    if (auto A = rc_recur fieldAlignment<model::StructType>(ABI, Type, Cache))
      Alignment = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeKind::UnionType:
    // The alignment of a union is the same as the alignment of its most
    // strictly aligned member.
    if (auto A = rc_recur fieldAlignment<model::UnionType>(ABI, Type, Cache))
      Alignment = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeKind::Invalid:
  case model::TypeKind::Count:
  default:
    revng_abort();
  }

  Cache[&Type] = Alignment;
  rc_return Alignment;
}

static RecursiveCoroutine<std::optional<uint64_t>>
naturalAlignment(const abi::Definition &ABI,
                 const model::QualifiedType &QT,
                 abi::Definition::AlignmentCache &Cache) {
  // This code assumes that the QualifiedType is well formed.
  for (auto It = QT.Qualifiers().begin(); It != QT.Qualifiers().end(); ++It) {
    switch (It->Kind()) {
    case model::QualifierKind::Pointer:
      // Doesn't matter what the type is, use alignment of the pointer.
      rc_return ABI.ScalarTypes().at(It->Size()).alignedAt();

    case model::QualifierKind::Array: {
      // The alignment of an array is the same as the alignment of its element.
      const model::QualifiedType Element{
        QT.UnqualifiedType(), { std::next(It), QT.Qualifiers().end() }
      };
      if (auto MaybeAlignment = rc_recur naturalAlignment(ABI, Element, Cache))
        rc_return *MaybeAlignment;
      else
        rc_return std::nullopt;
    }

    case model::QualifierKind::Const:
      // Const has no impact on alignment, look at the next qualifier.
      break;

    case model::QualifierKind::Invalid:
    case model::QualifierKind::Count:
    default:
      rc_return std::nullopt;
    }
  }

  rc_return rc_recur naturalAlignment(ABI, *QT.UnqualifiedType().get(), Cache);
}

std::optional<uint64_t> Definition::alignment(const model::QualifiedType &QT,
                                              AlignmentCache &Cache) const {
  std::optional<uint64_t> Result = naturalAlignment(*this, QT, Cache);
  if (Result.has_value() && Result.value() != 0)
    return Result.value();
  else
    return std::nullopt;
}

} // namespace abi
