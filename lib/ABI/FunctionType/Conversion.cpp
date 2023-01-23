/// \file Conversion.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ABI/Trait.h"
#include "revng/Model/Binary.h"

namespace abi::FunctionType {

template<std::size_t Size>
using RegisterArray = std::array<model::Register::Values, Size>;

template<model::Architecture::Values Architecture,
         typename RegisterType,
         size_t RegisterCount>
bool verify(const SortedVector<RegisterType> &UsedRegisters,
            const RegisterArray<RegisterCount> &AllowedRegisters) {
  for (const model::Register::Values &Register : AllowedRegisters) {
    // Verify the architecture of allowed registers.
    if (model::Register::getReferenceArchitecture(Register) != Architecture)
      revng_abort();

    // Verify that there are no duplicate allowed registers.
    if (llvm::count(AllowedRegisters, Register) != 1)
      revng_abort();
  }

  for (const RegisterType &Register : UsedRegisters) {
    // Verify the architecture of used registers.
    constexpr model::Architecture::Values A = Architecture;
    if (model::Register::getReferenceArchitecture(Register.Location()) != A)
      revng_abort();
  }

  // Verify that every used register is also allowed.
  for (const RegisterType &Register : UsedRegisters)
    if (llvm::count(AllowedRegisters, Register.Location()) != 1)
      return false;

  return true;
}

static std::optional<model::QualifiedType>
buildDoubleType(model::Register::Values UpperRegister,
                model::Register::Values LowerRegister,
                model::PrimitiveTypeKind::Values CustomKind,
                model::Binary &Binary) {
  model::PrimitiveTypeKind::Values UpperKind = selectTypeKind(UpperRegister);
  model::PrimitiveTypeKind::Values LowerKind = selectTypeKind(LowerRegister);
  if (UpperKind != LowerKind)
    return std::nullopt;

  size_t UpperSize = model::Register::getSize(UpperRegister);
  size_t LowerSize = model::Register::getSize(LowerRegister);
  return model::QualifiedType(Binary.getPrimitiveType(CustomKind,
                                                      UpperSize + LowerSize),
                              {});
}

static model::QualifiedType getTypeOrDefault(const model::QualifiedType &Type,
                                             model::Register::Values Register,
                                             model::Binary &Binary) {
  if (Type.UnqualifiedType().get() != nullptr)
    return Type;
  else
    return buildType(Register, Binary);
}

namespace ModelArch = model::Architecture;

template<model::ABI::Values ABI>
class ToCABIConverter {
  using AT = abi::Trait<ABI>;
  static constexpr auto Architecture = model::ABI::getArchitecture(ABI);
  static constexpr auto RegisterSize = ModelArch::getPointerSize(Architecture);

public:
  /// Entry point for the `toCABI` conversion.
  std::optional<model::TypePath>
  tryConvert(const model::RawFunctionType &Function,
             TupleTree<model::Binary> &Binary) const;

private:
  template<typename RegisterType, size_t RegisterCount, bool DryRun = false>
  std::optional<llvm::SmallVector<model::Argument, 8>>
  convertArguments(const SortedVector<RegisterType> &UsedRegisters,
                   const RegisterArray<RegisterCount> &AllowedRegisters,
                   model::Binary &Binary) const;

  llvm::SmallVector<model::Argument, 8>
  convertStackArguments(model::QualifiedType StackArgumentTypes,
                        size_t IndexOffset) const;

  template<typename RegisterType, size_t RegCount, bool DryRun = false>
  std::optional<model::QualifiedType>
  convertReturnValue(const SortedVector<RegisterType> &UsedRegisters,
                     const RegisterArray<RegCount> &AllowedRegisters,
                     model::Register::Values PointerToCopyLocation,
                     model::Binary &Binary) const;

  template<typename RType, size_t RCount>
  bool verifyArgumentsToBeConvertible(const SortedVector<RType> &UR,
                                      const RegisterArray<RCount> &AR,
                                      model::Binary &B) const {
    return convertArguments<RType, RCount, true>(UR, AR, B).has_value();
  }

  template<typename RType, size_t RCount>
  bool verifyReturnValueToBeConvertible(const SortedVector<RType> &UR,
                                        const RegisterArray<RCount> &AR,
                                        const model::Register::Values PtC,
                                        model::Binary &B) const {
    return convertReturnValue<RType, RCount, true>(UR, AR, PtC, B).has_value();
  }
};

template<model::ABI::Values ABI>
std::optional<model::TypePath>
ToCABIConverter<ABI>::tryConvert(const model::RawFunctionType &Function,
                                 TupleTree<model::Binary> &Binary) const {
  if (!verify<Architecture>(Function.Arguments(),
                            AT::GeneralPurposeArgumentRegisters))
    return std::nullopt;
  if (!verify<Architecture>(Function.ReturnValues(),
                            AT::GeneralPurposeReturnValueRegisters))
    return std::nullopt;

  // Verify the architecture of return value location register if present.
  constexpr model::Register::Values L = AT::ReturnValueLocationRegister;
  if (L != model::Register::Invalid)
    revng_assert(model::Register::getReferenceArchitecture(L) == Architecture);

  // Verify the architecture of callee saved registers.
  for (auto &R : AT::CalleeSavedRegisters) {
    revng_assert(model::Register::getReferenceArchitecture(R) == Architecture);
  }

  model::CABIFunctionType Result;
  Result.CustomName() = Function.CustomName();
  Result.OriginalName() = Function.OriginalName();
  Result.ABI() = ABI;

  if (!verifyArgumentsToBeConvertible(Function.Arguments(),
                                      AT::GeneralPurposeArgumentRegisters,
                                      *Binary))
    return std::nullopt;

  using C = AT;
  if (!verifyReturnValueToBeConvertible(Function.ReturnValues(),
                                        C::GeneralPurposeReturnValueRegisters,
                                        C::ReturnValueLocationRegister,
                                        *Binary))
    return std::nullopt;

  auto ArgumentList = convertArguments(Function.Arguments(),
                                       AT::GeneralPurposeArgumentRegisters,
                                       *Binary);
  revng_assert(ArgumentList != std::nullopt);
  for (auto &Argument : *ArgumentList)
    Result.Arguments().insert(Argument);

  auto StackArgumentList = convertStackArguments(Function.StackArgumentsType(),
                                                 Result.Arguments().size());
  for (auto &Argument : StackArgumentList)
    Result.Arguments().insert(Argument);

  auto ReturnValue = convertReturnValue(Function.ReturnValues(),
                                        C::GeneralPurposeReturnValueRegisters,
                                        C::ReturnValueLocationRegister,
                                        *Binary);
  revng_assert(ReturnValue != std::nullopt);
  Result.ReturnType() = *ReturnValue;

  // Steal the ID
  Result.ID() = Function.ID();

  // Add converted type to the model.
  using UT = model::UpcastableType;
  auto Ptr = UT::make<model::CABIFunctionType>(std::move(Result));
  auto NewTypePath = Binary->recordNewType(std::move(Ptr));

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceAllUsesWith(Function.key(), NewTypePath, Binary);

  // And don't forget to remove the old type.
  Binary->Types().erase(Function.key());

  return NewTypePath;
}

template<model::ABI::Values ABI>
using TCC = ToCABIConverter<ABI>;

template<model::ABI::Values ABI>
template<typename RegisterType, size_t RegisterCount, bool DryRun>
std::optional<llvm::SmallVector<model::Argument, 8>>
TCC<ABI>::convertArguments(const SortedVector<RegisterType> &UsedRegisters,
                           const RegisterArray<RegisterCount> &AllowedRegisters,
                           model::Binary &Binary) const {
  llvm::SmallVector<model::Argument, 8> Result;

  bool MustUseTheNextOne = false;
  auto AllowedRange = llvm::enumerate(llvm::reverse(AllowedRegisters));
  for (auto Pair : AllowedRange) {
    size_t Index = AllowedRegisters.size() - Pair.index() - 1;
    model::Register::Values Register = Pair.value();
    bool IsUsed = UsedRegisters.find(Register) != UsedRegisters.end();
    if (IsUsed) {
      model::Argument Temporary;
      if constexpr (!DryRun)
        Temporary.Type() = getTypeOrDefault(UsedRegisters.at(Register).Type(),
                                            Register,
                                            Binary);
      Temporary.CustomName() = UsedRegisters.at(Register).CustomName();
      Result.emplace_back(Temporary);
    } else if (MustUseTheNextOne) {
      if constexpr (!AT::OnlyStartDoubleArgumentsFromAnEvenRegister) {
        return std::nullopt;
      } else if ((Index & 1) == 0) {
        return std::nullopt;
      } else if (Result.size() > 1 && Index > 1) {
        auto &First = Result[Result.size() - 1];
        auto &Second = Result[Result.size() - 2];

        // TODO: see what can be done to preserve names better
        if (First.CustomName().empty() && !Second.CustomName().empty())
          First.CustomName() = Second.CustomName();

        if constexpr (!DryRun) {
          auto NewType = buildDoubleType(AllowedRegisters.at(Index - 2),
                                         AllowedRegisters.at(Index - 1),
                                         model::PrimitiveTypeKind::Generic,
                                         Binary);
          if (NewType == std::nullopt)
            return std::nullopt;

          First.Type() = *NewType;
        }

        Result.pop_back();
      } else {
        return std::nullopt;
      }
    }

    MustUseTheNextOne = MustUseTheNextOne || IsUsed;
  }

  for (auto Pair : llvm::enumerate(llvm::reverse(Result)))
    Pair.value().Index() = Pair.index();

  return Result;
}

template<model::ABI::Values ABI>
llvm::SmallVector<model::Argument, 8>
TCC<ABI>::convertStackArguments(model::QualifiedType StackArgumentTypes,
                                size_t IndexOffset) const {
  revng_assert(StackArgumentTypes.Qualifiers().empty());
  auto *Unqualified = StackArgumentTypes.UnqualifiedType().get();
  if (not Unqualified)
    return {};

  auto *Pointer = llvm::dyn_cast<model::StructType>(Unqualified);
  revng_assert(Pointer != nullptr,
               "`RawFunctionType::StackArgumentsType` must be a struct");
  const model::StructType &Types = *Pointer;

  llvm::SmallVector<model::Argument, 8> Result;
  for (const model::StructField &Field : Types.Fields()) {
    model::Argument &New = Result.emplace_back();
    New.Index() = IndexOffset++;
    New.Type() = Field.Type();
    New.CustomName() = Field.CustomName();
    New.OriginalName() = Field.OriginalName();
  }

  return Result;
}

template<model::ABI::Values ABI>
template<typename RegisterType, size_t RegCount, bool DryRun>
std::optional<model::QualifiedType>
TCC<ABI>::convertReturnValue(const SortedVector<RegisterType> &UsedRegisters,
                             const RegisterArray<RegCount> &AllowedRegisters,
                             model::Register::Values PointerToCopyLocation,
                             model::Binary &Binary) const {
  if (UsedRegisters.size() == 0) {
    auto Void = Binary.getPrimitiveType(model::PrimitiveTypeKind::Void, 0);
    return model::QualifiedType{ Void, {} };
  }

  if (UsedRegisters.size() == 1) {
    if (UsedRegisters.begin()->Location() == PointerToCopyLocation) {
      if constexpr (DryRun)
        return model::QualifiedType{};
      else
        return getTypeOrDefault(UsedRegisters.begin()->Type(),
                                PointerToCopyLocation,
                                Binary);
    } else {
      if constexpr (RegCount == 0)
        return std::nullopt;
      if (AllowedRegisters.front() == UsedRegisters.begin()->Location()) {
        if constexpr (DryRun)
          return model::QualifiedType{};
        else
          return getTypeOrDefault(UsedRegisters.begin()->Type(),
                                  UsedRegisters.begin()->Location(),
                                  Binary);
      } else {
        return std::nullopt;
      }
    }
  } else {
    model::UpcastableType Result = model::makeType<model::StructType>();
    auto ReturnStruct = llvm::dyn_cast<model::StructType>(Result.get());

    bool MustUseTheNextOne = false;
    auto AllowedRange = llvm::enumerate(llvm::reverse(AllowedRegisters));
    for (auto Pair : AllowedRange) {
      size_t Index = AllowedRegisters.size() - Pair.index() - 1;
      model::Register::Values Register = Pair.value();
      auto UsedIterator = UsedRegisters.find(Register);

      bool IsCurrentRegisterUsed = UsedIterator != UsedRegisters.end();
      if (IsCurrentRegisterUsed) {
        model::StructField CurrentField;
        CurrentField.Offset() = ReturnStruct->Size();
        if constexpr (!DryRun)
          CurrentField.Type() = getTypeOrDefault(UsedIterator->Type(),
                                                 UsedIterator->Location(),
                                                 Binary);
        ReturnStruct->Fields().insert(std::move(CurrentField));

        ReturnStruct->Size() += model::Register::getSize(Register);
      } else if (MustUseTheNextOne) {
        if constexpr (!AT::OnlyStartDoubleArgumentsFromAnEvenRegister)
          return std::nullopt;
        else if ((Index & 1) == 0 || ReturnStruct->Fields().size() <= 1
                 || Index <= 1)
          return std::nullopt;
      }

      MustUseTheNextOne = MustUseTheNextOne || IsCurrentRegisterUsed;
    }

    revng_assert(ReturnStruct->Size() != 0 && !ReturnStruct->Fields().empty());

    if constexpr (!DryRun) {
      auto ReturnStructTypePath = Binary.recordNewType(std::move(Result));
      revng_assert(ReturnStructTypePath.isValid());
      return model::QualifiedType{ ReturnStructTypePath, {} };
    } else {
      return model::QualifiedType{};
    }
  }

  return std::nullopt;
}

std::optional<model::TypePath>
tryConvertToCABI(const model::RawFunctionType &Function,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();
  revng_assert(*MaybeABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(*MaybeABI, [&]<model::ABI::Values A>() {
    return ToCABIConverter<A>().tryConvert(Function, Binary);
  });
}

} // namespace abi::FunctionType
