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
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/YAMLTraits.h"

namespace {

template<std::ranges::range RegisterContainer>
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

bool isVectorRegister(model::Register::Values Register) {
  using model::Register::primitiveKind;
  return primitiveKind(Register) == model::PrimitiveKind::Float;
}

/// Helps detecting unsupported ABI trait definition with respect to
/// the way they return the return values.
///
/// This is an important piece of abi trait verification. For more information
/// see the `static_assert` that invokes it in \ref distributeArguments
///
/// \return `true` if the ABI is valid, `false` otherwise.
bool verifyReturnValueLocation(const abi::Definition &D) {
  if (not model::Register::isValid(D.ReturnValueLocationRegister())) {
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

abi::ScalarKind::Values getScalarKind(abi::CType::Values Type) {
  if (abi::CType::Char <= Type and Type <= abi::CType::LongLong)
    return abi::ScalarKind::Integer;

  if (abi::CType::Float <= Type and Type <= abi::CType::LongDouble)
    return abi::ScalarKind::FloatingPoint;

  return abi::ScalarKind::Invalid;
}

} // namespace

namespace abi {

const ScalarType *Definition::findIntegerType(uint64_t Size) const {
  // TODO: If and when invalidation tracking is introduced for ABI definitions,
  //       this function should be changed to use tryGet in place of find.
  auto I = ScalarTypes().find({ ScalarKind::Integer, Size });
  return I != ScalarTypes().end() ? &*I : nullptr;
}

const ScalarType *Definition::findFloatingPointType(uint64_t Size) const {
  // TODO: If and when invalidation tracking is introduced for ABI definitions,
  //       this function should be changed to use tryGet in place of find.
  auto I = ScalarTypes().find({ ScalarKind::FloatingPoint, Size });
  return I != ScalarTypes().end() ? &*I : nullptr;
}

const ScalarType &Definition::getWidestIntegerType() const {
  for (const ScalarType &Type : llvm::reverse(ScalarTypes())) {
    if (Type.Kind() == ScalarKind::Integer)
      return Type;
  }
  revng_abort("Invalid ABI definition.");
}

bool Definition::verify() const {
  if (not model::ABI::isValid(ABI()))
    return false;

  if (StackAlignment() == 0)
    return false;

  if (MinimumStackArgumentSize() == 0)
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

  bool CTypes[CType::Count] = {};
  bool HasIntegerScalar = false;

  for (const abi::ScalarType &Type : ScalarTypes()) {
    if (Type.Kind() == ScalarKind::Invalid)
      return false;

    if (Type.Kind() == ScalarKind::Integer)
      HasIntegerScalar = true;

    if (Type.Size() == 0)
      return false;

    for (auto CT : Type.CTypes()) {
      if (CT == CType::Invalid)
        return false;

      // The C type kind must match the scalar kind.
      if (getScalarKind(CT) != Type.Kind())
        return false;

      // The ABI may not associate the same C type with multiple scalars.
      if (std::exchange(CTypes[CT], true))
        return false;
    }
  }

  // The ABI must define at least one integer scalar.
  if (not HasIntegerScalar)
    return false;

  if (not getDataModel().verify())
    return false;

  return true;
}

using RFT = model::RawFunctionDefinition;
bool Definition::isPreliminarilyCompatibleWith(const RFT &Function) const {
  revng_assert(verify());
  const auto Architecture = model::ABI::getRegisterArchitecture(ABI());

  SortedVector<model::Register::Values> Arguments;
  for (auto I = Arguments.batch_insert(); auto R : Function.Arguments()) {
    if (!model::Register::isUsedInArchitecture(R.Location(), Architecture))
      return false;

    I.emplace(R.Location());
  }

  SortedVector<model::Register::Values> AllowedArguments;
  {
    auto I = AllowedArguments.batch_insert_or_assign();
    for (model::Register::Values R : GeneralPurposeArgumentRegisters())
      I.emplace_or_assign(R);
    for (model::Register::Values R : VectorArgumentRegisters())
      I.emplace_or_assign(R);
  }

  if (!std::includes(AllowedArguments.begin(),
                     AllowedArguments.end(),
                     Arguments.begin(),
                     Arguments.end())) {
    return false;
  }

  SortedVector<model::Register::Values> ReturnValues;
  for (auto I = ReturnValues.batch_insert(); auto R : Function.ReturnValues()) {
    if (!model::Register::isUsedInArchitecture(R.Location(), Architecture))
      return false;

    I.emplace(R.Location());
  }

  SortedVector<model::Register::Values> AllowedReturnValues;
  {
    auto I = AllowedReturnValues.batch_insert_or_assign();
    for (model::Register::Values R : GeneralPurposeReturnValueRegisters())
      I.emplace_or_assign(R);
    for (model::Register::Values R : VectorReturnValueRegisters())
      I.emplace_or_assign(R);
  }

  if (!std::includes(AllowedReturnValues.begin(),
                     AllowedReturnValues.end(),
                     ReturnValues.begin(),
                     ReturnValues.end())) {
    return false;
  }

  for (model::Register::Values Register : Function.PreservedRegisters())
    if (!model::Register::isUsedInArchitecture(Register, Architecture))
      return false;

  return true;
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
    std::string Error = "The ABI definition is missing for: " + ::toString(ABI);
    revng_abort(Error.c_str());
  }

  auto Parsed = TupleTree<Definition>::fromFile(MaybePath.value());
  if (!Parsed) {
    std::string Error = "Unable to deserialize the definition for: "
                        + ::toString(ABI);
    revng_abort(Error.c_str());
  }

  if (!Parsed->verify()) {
    std::string Error = "Deserialized ABI definition is not valid: "
                        + ::toString(ABI);
    revng_abort(Error.c_str());
  }

  auto &&[It, Success] = DefinitionCache.try_emplace(ABI, std::move(**Parsed));
  revng_assert(Success);
  return It->second;
}

using AlignmentInfo = abi::Definition::AlignmentInfo;
static RecursiveCoroutine<std::optional<AlignmentInfo>>
naturalAlignment(const abi::Definition &ABI,
                 const model::Type &Type,
                 abi::Definition::AlignmentCache &Cache);
using AlignmentInfo = abi::Definition::AlignmentInfo;
static RecursiveCoroutine<std::optional<AlignmentInfo>>
naturalAlignment(const abi::Definition &ABI,
                 const model::TypeDefinition &Type,
                 abi::Definition::AlignmentCache &Cache);

template<typename RealType>
RecursiveCoroutine<std::optional<AlignmentInfo>>
underlyingAlignment(const abi::Definition &ABI,
                    const model::TypeDefinition &Type,
                    abi::Definition::AlignmentCache &Cache) {
  const auto &Underlying = llvm::cast<RealType>(Type).UnderlyingType();
  rc_return rc_recur naturalAlignment(ABI, *Underlying, Cache);
}

template<typename RealType>
RecursiveCoroutine<std::optional<AlignmentInfo>>
fieldAlignment(const abi::Definition &ABI,
               const model::TypeDefinition &Type,
               abi::Definition::AlignmentCache &Cache) {
  AlignmentInfo Result = { 1, true };
  for (const auto &Field : llvm::cast<RealType>(Type).Fields()) {
    if (auto A = rc_recur naturalAlignment(ABI, *Field.Type(), Cache)) {
      Result.Value = std::max(Result.Value, A->Value);
      Result.IsNatural = Result.IsNatural && A->IsNatural;
      if (Result.IsNatural)
        if constexpr (std::is_same_v<RealType, model::StructDefinition>)
          if (Field.Offset() % A->Value != 0)
            Result.IsNatural = false;
    } else {
      rc_return std::nullopt;
    }
  }

  rc_return Result;
}

static RecursiveCoroutine<std::optional<AlignmentInfo>>
naturalAlignment(const abi::Definition &ABI,
                 const model::TypeDefinition &Type,
                 abi::Definition::AlignmentCache &Cache) {
  if (auto Iterator = Cache.find(&Type); Iterator != Cache.end())
    rc_return Iterator->second;

  AlignmentInfo Result = { 0, true };

  // This code assumes that the type `Type` is well formed.
  switch (Type.Kind()) {
  case model::TypeDefinitionKind::RawFunctionDefinition:
  case model::TypeDefinitionKind::CABIFunctionDefinition:
    // Function prototypes have no size - hence no alignment.
    rc_return std::nullopt;

  case model::TypeDefinitionKind::EnumDefinition:
    // The alignment of an enum is the same as the alignment of its underlying
    // type
    using modelEnumType = model::EnumDefinition;
    if (auto A = rc_recur underlyingAlignment<modelEnumType>(ABI, Type, Cache))
      Result = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeDefinitionKind::TypedefDefinition:
    // The alignment of an enum is the same as the alignment of its underlying
    // type
    using TypedefD = model::TypedefDefinition;
    if (auto A = rc_recur underlyingAlignment<TypedefD>(ABI, Type, Cache))
      Result = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeDefinitionKind::StructDefinition:
    // The alignment of a struct is the same as the alignment of its most
    // strictly aligned member.
    using StructD = model::StructDefinition;
    if (auto A = rc_recur fieldAlignment<StructD>(ABI, Type, Cache))
      Result = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeDefinitionKind::UnionDefinition:
    // The alignment of a union is the same as the alignment of its most
    // strictly aligned member.
    using UnionD = model::UnionDefinition;
    if (auto A = rc_recur fieldAlignment<UnionD>(ABI, Type, Cache))
      Result = *A;
    else
      rc_return std::nullopt;
    break;

  case model::TypeDefinitionKind::Invalid:
  case model::TypeDefinitionKind::Count:
  default:
    revng_abort();
  }

  Cache[&Type] = Result;
  rc_return Result;
}

static RecursiveCoroutine<std::optional<AlignmentInfo>>
naturalAlignment(const abi::Definition &ABI,
                 const model::Type &Type,
                 abi::Definition::AlignmentCache &Cache) {
  if (const auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    // The alignment of an array is the same as the alignment of its element.
    rc_return rc_recur naturalAlignment(ABI, *Array->ElementType(), Cache);

  } else if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    rc_return rc_recur naturalAlignment(ABI, D->unwrap(), Cache);

  } else if (const auto *P = llvm::dyn_cast<model::PointerType>(&Type)) {
    // Doesn't matter what the type is, use alignment of the pointer.
    rc_return AlignmentInfo{ ABI.findIntegerType(P->PointerSize())->alignedAt(),
                             true };

  } else if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    // The alignment of primitives is easy to figure out based on the abi.
    if (P->PrimitiveKind() == model::PrimitiveKind::Void) {
      // `void` has no size - hence no alignment.
      revng_assert(P->Size() == 0);

      rc_return AlignmentInfo{ 0, false };
    } else if (P->PrimitiveKind() == model::PrimitiveKind::Float) {
      if (auto ABIType = ABI.findFloatingPointType(P->Size()))
        rc_return AlignmentInfo{ ABIType->alignedAt(), true };

      rc_return std::nullopt;
    } else {
      if (auto ABIType = ABI.findIntegerType(P->Size()))
        rc_return AlignmentInfo{ ABIType->alignedAt(), true };

      rc_return std::nullopt;
    }
  } else {
    revng_abort("Unsupported type.");
  }
}

template<Yamlizable T>
std::optional<AlignmentInfo>
assertOnFailure(std::optional<AlignmentInfo> &&ComputationResult,
                const T &ThingToDumpOnFailure) {
  if (!ComputationResult) {
    std::string Error = "Unable to compute the alignment of "
                        + toString(ThingToDumpOnFailure);
    revng_abort(Error.c_str());
  }

  return std::move(ComputationResult);
}

std::optional<uint64_t> Definition::alignment(const model::Type &Type,
                                              AlignmentCache &Cache) const {
  auto Result = assertOnFailure(naturalAlignment(*this, Type, Cache),
                                model::copyType(Type));
  if (Result->Value == 0)
    return std::nullopt;

  return Result->IsNatural ? Result->Value : 1;
}
std::optional<uint64_t> Definition::alignment(const model::TypeDefinition &Type,
                                              AlignmentCache &Cache) const {
  auto Result = assertOnFailure(naturalAlignment(*this, Type, Cache),
                                model::copyTypeDefinition(Type));
  if (Result->Value == 0)
    return std::nullopt;

  return Result->IsNatural ? Result->Value : 1;
}

std::optional<bool>
Definition::hasNaturalAlignment(const model::Type &Type,
                                AlignmentCache &Cache) const {
  auto Result = assertOnFailure(naturalAlignment(*this, Type, Cache),
                                model::copyType(Type));
  if (Result->Value == 0)
    return std::nullopt;

  return Result->IsNatural;
}
std::optional<bool>
Definition::hasNaturalAlignment(const model::TypeDefinition &Type,
                                AlignmentCache &Cache) const {
  auto Result = assertOnFailure(naturalAlignment(*this, Type, Cache),
                                model::copyTypeDefinition(Type));
  if (Result->Value == 0)
    return std::nullopt;

  return Result->IsNatural;
}

CDataModel Definition::getDataModel() const {
  CDataModel DM = CDataModel::getDefaultDataModel(getPointerSize());

  for (const ScalarType &Type : ScalarTypes()) {
    for (auto CT : Type.CTypes()) {
      switch (CT) {
      case CType::Char:
        DM.getStandardTypeSize(CStandardType::Char) = Type.Size();
        break;
      case CType::Short:
        DM.getStandardTypeSize(CStandardType::Short) = Type.Size();
        break;
      case CType::Int:
        DM.getStandardTypeSize(CStandardType::Int) = Type.Size();
        break;
      case CType::Long:
        DM.getStandardTypeSize(CStandardType::Long) = Type.Size();
        break;
      case CType::LongLong:
        DM.getStandardTypeSize(CStandardType::LongLong) = Type.Size();
        break;
      case CType::Float:
        DM.getStandardTypeSize(CStandardType::Float) = Type.Size();
        break;
      case CType::Double:
        DM.getStandardTypeSize(CStandardType::Double) = Type.Size();
        break;
      case CType::LongDouble:
        DM.getStandardTypeSize(CStandardType::LongDouble) = Type.Size();
        break;
      case CType::Invalid:
      case CType::Count:
        revng_abort("Invalid C type in ABI definition.");
      }
    }
  }

  return DM;
}

} // namespace abi
