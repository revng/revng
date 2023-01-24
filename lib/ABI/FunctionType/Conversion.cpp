/// \file Conversion.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ABI/FunctionType/TypeBucket.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"

namespace abi::FunctionType {

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

class ToCABIConverter {
private:
  using ArgumentRegisters = SortedVector<model::NamedTypedRegister>;
  using ReturnValueRegisters = SortedVector<model::TypedRegister>;

public:
  struct Converted {
    llvm::SmallVector<model::Argument, 8> RegisterArguments;
    llvm::SmallVector<model::Argument, 8> StackArguments;
    model::QualifiedType ReturnValueType;
  };

public:
  std::optional<Converted> Result;

private:
  const abi::Definition &ABI;
  model::Binary &Binary;
  TypeBucket Bucket;

public:
  ToCABIConverter(const abi::Definition &ABI, model::Binary &Binary) :
    Result(std::nullopt), ABI(ABI), Binary(Binary), Bucket(Binary) {}

  std::optional<Converted>
  tryConvert(const model::RawFunctionType &FunctionType) {
    revng_assert(Bucket.empty());

    // Register arguments first.
    auto Arguments = tryConvertingRegisterArguments(FunctionType.Arguments());
    if (!Arguments.has_value()) {
      Bucket.drop();
      return std::nullopt;
    }

    // Then stack ones.
    auto Stack = tryConvertingStackArguments(FunctionType.StackArgumentsType(),
                                             Arguments->size());
    if (!Stack.has_value()) {
      Bucket.drop();
      return std::nullopt;
    }

    // And the return value.
    auto ReturnType = tryConvertingReturnValue(FunctionType.ReturnValues());
    if (!ReturnType.has_value()) {
      Bucket.drop();
      return std::nullopt;
    }

    // Conversion successful, return the result.
    Bucket.commit();
    return Converted{ std::move(*Arguments),
                      std::move(*Stack),
                      std::move(*ReturnType) };
  }

private:
  /// Helper used for converting register arguments to the c-style
  /// representation
  ///
  /// \param Registers a set of registers confirmed to be in use by
  ///        the function in question.
  ///
  /// \return a list of arguments if the conversion was successful,
  ///         `std::nullopt` otherwise.
  std::optional<llvm::SmallVector<model::Argument, 8>>
  tryConvertingRegisterArguments(const ArgumentRegisters &Registers);

  /// Helper used for converting stack argument struct into
  /// the c-style representation
  ///
  /// \param StackArgumentTypes The qualified type of the relevant part of
  ///        the stack.
  /// \param IndexOffset The index of the first argument (should be set to
  ///        the number of register arguments).
  ///
  /// \return An ordered list of arguments.
  std::optional<llvm::SmallVector<model::Argument, 8>>
  tryConvertingStackArguments(model::QualifiedType StackArgumentTypes,
                              size_t IndexOffset);

  /// Helper used for converting return values to the c-style representation
  ///
  /// \param Registers a set of registers confirmed to be in use by
  ///        the function in question.
  /// \param ReturnValueLocation an optional register used for pointing to
  ///        return values in memory on some architectures. It is set to
  ///        `nullptr` if it's not applicable.
  ///
  /// \return a qualified type if conversion is possible, `std::nullopt`
  ///         otherwise.
  std::optional<model::QualifiedType>
  tryConvertingReturnValue(const ReturnValueRegisters &Registers);
};

std::optional<model::TypePath>
tryConvertToCABI(const model::RawFunctionType &FunctionType,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();

  const abi::Definition &ABI = abi::Definition::get(*MaybeABI);
  if (ABI.isIncompatibleWith(FunctionType))
    return std::nullopt;

  ToCABIConverter Converter(ABI, *Binary);
  std::optional Converted = Converter.tryConvert(FunctionType);
  if (!Converted.has_value())
    return std::nullopt;

  // The conversion was successful, a new `CABIFunctionType` can now be created,
  auto [NewType, NewTypePath] = Binary->makeType<model::CABIFunctionType>();
  model::copyMetadata(NewType, FunctionType);
  NewType.ABI() = ABI.ABI();

  // And filled in with the argument information.
  auto Arguments = llvm::concat<model::Argument>(Converted->RegisterArguments,
                                                 Converted->StackArguments);
  for (auto &Argument : Arguments)
    NewType.Arguments().insert(std::move(Argument));
  NewType.ReturnType() = Converted->ReturnValueType;

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceAllUsesWith(FunctionType.key(), NewTypePath, Binary);

  // And don't forget to remove the old type.
  Binary->Types().erase(FunctionType.key());

  return NewTypePath;
}

using TCC = ToCABIConverter;
std::optional<llvm::SmallVector<model::Argument, 8>>
TCC::tryConvertingRegisterArguments(const ArgumentRegisters &Registers) {
  constexpr bool DryRun = false;
  llvm::SmallVector<model::Argument, 8> Result;

  const auto &AllowedRegisters = ABI.GeneralPurposeArgumentRegisters();

  bool MustUseTheNextOne = false;
  auto AllowedRange = llvm::enumerate(llvm::reverse(AllowedRegisters));
  for (auto Pair : AllowedRange) {
    size_t Index = AllowedRegisters.size() - Pair.index() - 1;
    model::Register::Values Register = Pair.value();
    bool IsUsed = Registers.find(Register) != Registers.end();
    if (IsUsed) {
      model::Argument Temporary;
      if constexpr (!DryRun)
        Temporary.Type() = getTypeOrDefault(Registers.at(Register).Type(),
                                            Register,
                                            Binary);
      Temporary.CustomName() = Registers.at(Register).CustomName();
      Result.emplace_back(Temporary);
    } else if (MustUseTheNextOne) {
      if (!ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
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

std::optional<llvm::SmallVector<model::Argument, 8>>
TCC::tryConvertingStackArguments(model::QualifiedType StackArgumentTypes,
                                 size_t IndexOffset) {
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

std::optional<model::QualifiedType>
TCC::tryConvertingReturnValue(const ReturnValueRegisters &Registers) {
  constexpr bool DryRun = false;
  const auto &AllowedRegisters = ABI.GeneralPurposeReturnValueRegisters();
  const auto &ReturnValueLocationRegister = ABI.ReturnValueLocationRegister();

  if (Registers.size() == 0) {
    auto Void = Binary.getPrimitiveType(model::PrimitiveTypeKind::Void, 0);
    return model::QualifiedType{ Void, {} };
  }

  if (Registers.size() == 1) {
    if (Registers.begin()->Location() == ReturnValueLocationRegister) {
      if constexpr (DryRun)
        return model::QualifiedType{};
      else
        return getTypeOrDefault(Registers.begin()->Type(),
                                ReturnValueLocationRegister,
                                Binary);
    } else {
      if (AllowedRegisters.size() == 0)
        return std::nullopt;
      if (AllowedRegisters.front() == Registers.begin()->Location()) {
        if constexpr (DryRun)
          return model::QualifiedType{};
        else
          return getTypeOrDefault(Registers.begin()->Type(),
                                  Registers.begin()->Location(),
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
      auto UsedIterator = Registers.find(Register);

      bool IsCurrentRegisterUsed = UsedIterator != Registers.end();
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
        if (!ABI.OnlyStartDoubleArgumentsFromAnEvenRegister())
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

} // namespace abi::FunctionType
