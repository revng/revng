/// \file Conversion.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/Model/Binary.h"

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

private:
  const abi::Definition &ABI;

public:
  explicit ToCABIConverter(const abi::Definition &ABI) : ABI(ABI) {
    ABI.verify();
  }

public:
  /// Entry point for the `toCABI` conversion.
  std::optional<model::TypePath>
  tryConvert(const model::RawFunctionType &Function,
             TupleTree<model::Binary> &Binary) const;

private:
  template<bool DryRun = false>
  std::optional<llvm::SmallVector<model::Argument, 8>>
  convertRegisterArguments(const ArgumentRegisters &Registers,
                           model::Binary &Binary) const;

  llvm::SmallVector<model::Argument, 8>
  convertStackArguments(model::QualifiedType StackArgumentTypes,
                        size_t IndexOffset) const;

  template<bool DryRun = false>
  std::optional<model::QualifiedType>
  convertReturnValue(const ReturnValueRegisters &Registers,
                     model::Binary &Binary) const;

  bool verifyArgumentsToBeConvertible(const ArgumentRegisters &Registers,
                                      model::Binary &Binary) const {
    return convertRegisterArguments<true>(Registers, Binary).has_value();
  }

  bool verifyReturnValueToBeConvertible(const ReturnValueRegisters &Registers,
                                        model::Binary &Binary) const {
    return convertReturnValue<true>(Registers, Binary).has_value();
  }
};

std::optional<model::TypePath>
ToCABIConverter::tryConvert(const model::RawFunctionType &Function,
                            TupleTree<model::Binary> &Binary) const {
  model::CABIFunctionType Result;
  Result.CustomName() = Function.CustomName();
  Result.OriginalName() = Function.OriginalName();
  Result.ABI() = ABI.ABI();

  if (!verifyArgumentsToBeConvertible(Function.Arguments(), *Binary))
    return std::nullopt;

  if (!verifyReturnValueToBeConvertible(Function.ReturnValues(), *Binary))
    return std::nullopt;

  auto ArgumentList = convertRegisterArguments(Function.Arguments(), *Binary);
  revng_assert(ArgumentList != std::nullopt);
  for (auto &Argument : *ArgumentList)
    Result.Arguments().insert(Argument);

  auto StackArgumentList = convertStackArguments(Function.StackArgumentsType(),
                                                 Result.Arguments().size());
  for (auto &Argument : StackArgumentList)
    Result.Arguments().insert(Argument);

  auto ReturnValue = convertReturnValue(Function.ReturnValues(), *Binary);
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

template<bool DryRun>
std::optional<llvm::SmallVector<model::Argument, 8>>
ToCABIConverter::convertRegisterArguments(const ArgumentRegisters &Registers,
                                          model::Binary &Binary) const {
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

llvm::SmallVector<model::Argument, 8>
ToCABIConverter::convertStackArguments(model::QualifiedType StackArgumentTypes,
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

template<bool DryRun>
std::optional<model::QualifiedType>
ToCABIConverter::convertReturnValue(const ReturnValueRegisters &Registers,
                                    model::Binary &Binary) const {
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

std::optional<model::TypePath>
tryConvertToCABI(const model::RawFunctionType &Function,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();

  ToCABIConverter ToCABI(abi::Definition::get(*MaybeABI));
  return ToCABI.tryConvert(Function, Binary);
}

} // namespace abi::FunctionType
