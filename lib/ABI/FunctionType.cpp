/// \file FunctionType.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "revng/ABI/FunctionType.h"
#include "revng/ABI/RegisterStateDeductions.h"
#include "revng/ABI/Trait.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Register.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/EnumSwitch.h"

namespace abi::FunctionType {

template<size_t Size>
using RegisterArray = std::array<model::Register::Values, Size>;

template<model::Architecture::Values Architecture,
         typename RegisterType,
         size_t RegisterCount>
bool verify(const SortedVector<RegisterType> &UsedRegisters,
            const RegisterArray<RegisterCount> &AllowedRegisters) {
  for (const model::Register::Values &Register : AllowedRegisters) {
    // Verify the architecture of allowed registers.
    if (model::Register::getArchitecture(Register) != Architecture)
      revng_abort();

    // Verify that there are no duplicate allowed registers.
    if (llvm::count(AllowedRegisters, Register) != 1)
      revng_abort();
  }

  for (const RegisterType &Register : UsedRegisters) {
    // Verify the architecture of used registers.
    if (model::Register::getArchitecture(Register.Location) != Architecture)
      revng_abort();
  }

  // Verify that every used register is also allowed.
  for (const RegisterType &Register : UsedRegisters)
    if (llvm::count(AllowedRegisters, Register.Location) != 1)
      return false;

  return true;
}

constexpr static model::PrimitiveTypeKind::Values
selectTypeKind(model::Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return model::PrimitiveTypeKind::PointerOrNumber;
}

static model::QualifiedType
buildType(model::Register::Values Register, model::Binary &TheBinary) {
  model::PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(TheBinary.getPrimitiveType(Kind, Size), {});
}

static model::QualifiedType
buildGenericType(model::Register::Values Register, model::Binary &TheBinary) {
  constexpr auto Kind = model::PrimitiveTypeKind::Generic;
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(TheBinary.getPrimitiveType(Kind, Size), {});
}

static std::optional<model::QualifiedType>
buildDoubleType(model::Register::Values UpperRegister,
                model::Register::Values LowerRegister,
                model::PrimitiveTypeKind::Values CustomKind,
                model::Binary &TheBinary) {
  model::PrimitiveTypeKind::Values UpperKind = selectTypeKind(UpperRegister);
  model::PrimitiveTypeKind::Values LowerKind = selectTypeKind(LowerRegister);
  if (UpperKind != LowerKind)
    return std::nullopt;

  size_t UpperSize = model::Register::getSize(UpperRegister);
  size_t LowerSize = model::Register::getSize(LowerRegister);
  return model::QualifiedType(TheBinary.getPrimitiveType(CustomKind,
                                                         UpperSize + LowerSize),
                              {});
}

static model::QualifiedType getTypeOrDefault(const model::QualifiedType &Type,
                                             model::Register::Values Register,
                                             model::Binary &Binary) {
  if (Type.UnqualifiedType.get() != nullptr)
    return Type;
  else
    return buildType(Register, Binary);
}

template<model::ABI::Values ABI>
class ConversionHelper {
  using AT = abi::Trait<ABI>;

  using IndexType = decltype(model::Argument::Index);
  using RegisterList = llvm::SmallVector<model::Register::Values, 1>;
  struct DistributedArgument {
    RegisterList Registers = {};
    size_t Size = 0, SizeOnStack = 0;
  };
  using DistributedArguments = llvm::SmallVector<DistributedArgument, 4>;

  using ArgumentContainer = SortedVector<model::Argument>;

public:
  static std::optional<model::CABIFunctionType>
  toCABI(const model::RawFunctionType &Function, model::Binary &TheBinary) {
    static constexpr auto Arch = model::ABI::getArchitecture(ABI);
    if (!verify<Arch>(Function.Arguments, AT::GeneralPurposeArgumentRegisters))
      return std::nullopt;
    if (!verify<Arch>(Function.ReturnValues,
                      AT::GeneralPurposeReturnValueRegisters))
      return std::nullopt;

    // Verify the architecture of return value location register if present.
    constexpr model::Register::Values PTCRR = AT::ReturnValueLocationRegister;
    if (PTCRR != model::Register::Invalid)
      revng_assert(model::Register::getArchitecture(PTCRR) == Arch);

    // Verify the architecture of callee saved registers.
    for (auto &SavedRegister : AT::CalleeSavedRegisters)
      revng_assert(model::Register::getArchitecture(SavedRegister) == Arch);

    model::CABIFunctionType Result;
    Result.CustomName = Function.CustomName;
    Result.ABI = ABI;

    if (!verifyArgumentsToBeConvertible(Function.Arguments,
                                        AT::GeneralPurposeArgumentRegisters,
                                        TheBinary))
      return std::nullopt;

    using C = AT;
    if (!verifyReturnValueToBeConvertible(Function.ReturnValues,
                                          C::GeneralPurposeReturnValueRegisters,
                                          C::ReturnValueLocationRegister,
                                          TheBinary))
      return std::nullopt;

    auto ArgumentList = convertArguments(Function.Arguments,
                                         AT::GeneralPurposeArgumentRegisters,
                                         TheBinary);
    revng_assert(ArgumentList != std::nullopt);
    for (auto &Argument : *ArgumentList)
      Result.Arguments.insert(Argument);

    auto ReturnValue = convertReturnValue(Function.ReturnValues,
                                          C::GeneralPurposeReturnValueRegisters,
                                          C::ReturnValueLocationRegister,
                                          TheBinary);
    revng_assert(ReturnValue != std::nullopt);
    Result.ReturnType = *ReturnValue;
    return Result;
  }

  static model::RawFunctionType
  toRaw(const model::CABIFunctionType &Function, model::Binary &TheBinary) {
    auto Arguments = distributeArguments(Function.Arguments);

    model::RawFunctionType Result;
    Result.CustomName = Function.CustomName;

    for (size_t ArgIndex = 0; ArgIndex < Arguments.size(); ++ArgIndex) {
      auto &ArgumentStorage = Arguments[ArgIndex];
      const auto &ArgumentType = Function.Arguments.at(ArgIndex).Type;
      if (!ArgumentStorage.Registers.empty()) {
        // Handle the registers
        auto OriginalName = Function.Arguments.at(ArgIndex).CustomName;
        for (size_t Index = 0; auto Register : ArgumentStorage.Registers) {
          auto FinalName = OriginalName;
          if (ArgumentStorage.Registers.size() > 1 && !FinalName.empty())
            FinalName += "_part_" + std::to_string(++Index) + "_out_of_"
                         + std::to_string(ArgumentStorage.Registers.size());
          model::NamedTypedRegister Argument(Register);
          Argument.Type = chooseArgumentType(ArgumentType,
                                             Register,
                                             ArgumentStorage.Registers,
                                             TheBinary);
          Argument.CustomName = FinalName;
          Result.Arguments.insert(Argument);
        }
      }

      if (ArgumentStorage.SizeOnStack != 0) {
        // Handle the stack
        auto ArgumentIterator = Function.Arguments.find(ArgIndex);
        revng_assert(ArgumentIterator != Function.Arguments.end());
        const model::Argument &Argument = *ArgumentIterator;

        /// TODO: handle stack arguments properly.
        /// \note: different ABIs could use different stack types.
        /// \sa: `clrcall` ABI.
      }
    }

    if (!Function.ReturnType.isVoid()) {
      auto ReturnValue = distributeReturnValue(Function.ReturnType);
      if (!ReturnValue.Registers.empty()) {
        // Handle a register-based return value.
        for (model::Register::Values Register : ReturnValue.Registers) {
          model::TypedRegister ReturnValueRegister;
          ReturnValueRegister.Location = Register;
          ReturnValueRegister.Type = chooseArgumentType(Function.ReturnType,
                                                        Register,
                                                        ReturnValue.Registers,
                                                        TheBinary);

          Result.ReturnValues.insert(std::move(ReturnValueRegister));
        }

        // Try and recover types from the struct if possible
        if (Function.ReturnType.Qualifiers.empty()) {
          const model::Type *Type = Function.ReturnType.UnqualifiedType.get();
          revng_assert(Type != nullptr);
          const auto *Struct = llvm::dyn_cast<model::StructType>(Type);
          if (Struct && Struct->Fields.size() == Result.ReturnValues.size()) {
            using RegisterEnum = model::Register::Values;
            SmallMap<RegisterEnum, model::QualifiedType, 4> RecoveredTypes;
            size_t StructOffset = 0;
            for (size_t Index = 0; Index < Struct->Fields.size(); ++Index) {
              if (Index >= AT::GeneralPurposeReturnValueRegisters.size())
                break;
              auto Register = AT::GeneralPurposeReturnValueRegisters[Index];

              auto TypedRegisterIterator = Result.ReturnValues.find(Register);
              if (TypedRegisterIterator == Result.ReturnValues.end())
                break;

              const model::StructField &Field = Struct->Fields.at(StructOffset);

              auto MaybeFieldSize = Field.Type.size();
              revng_assert(MaybeFieldSize != std::nullopt);

              auto MaybeRegisterSize = TypedRegisterIterator->Type.size();
              revng_assert(MaybeRegisterSize != std::nullopt);

              if (MaybeFieldSize.value() != MaybeRegisterSize.value())
                break;

              auto Tie = std::tie(Register, Field.Type);
              auto [Iterator, Success] = RecoveredTypes.insert(std::move(Tie));
              revng_assert(Success);

              StructOffset += MaybeFieldSize.value();
            }

            if (RecoveredTypes.size() == Result.ReturnValues.size())
              for (auto [Register, Type] : RecoveredTypes)
                Result.ReturnValues.at(Register).Type = Type;
          }
        }
      } else {
        // Handle a pointer-based return value.
        revng_assert(!AT::GeneralPurposeReturnValueRegisters.empty());
        auto Register = AT::GeneralPurposeReturnValueRegisters[0];
        auto RegisterSize = model::Register::getSize(Register);
        auto PointerQualifier = model::Qualifier::createPointer(RegisterSize);

        auto MaybeReturnValueSize = Function.ReturnType.size();
        revng_assert(MaybeReturnValueSize != std::nullopt);
        revng_assert(ReturnValue.Size == *MaybeReturnValueSize);

        model::QualifiedType ReturnType = Function.ReturnType;
        ReturnType.Qualifiers.emplace_back(PointerQualifier);

        model::TypedRegister ReturnPointer(Register);
        ReturnPointer.Type = std::move(ReturnType);
        Result.ReturnValues.insert(std::move(ReturnPointer));
      }
    }

    // Populate the list of preserved registers
    for (model::Register::Values Register : AT::CalleeSavedRegisters)
      Result.PreservedRegisters.insert(Register);

    return Result;
  }

private:
  template<typename RegisterType, size_t RegisterCount, bool DryRun = false>
  static std::optional<llvm::SmallVector<model::Argument, 8>>
  convertArguments(const SortedVector<RegisterType> &UsedRegisters,
                   const RegisterArray<RegisterCount> &AllowedRegisters,
                   model::Binary &TheBinary) {
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
          Temporary.Type = getTypeOrDefault(UsedRegisters.at(Register).Type,
                                            Register,
                                            TheBinary);
        Temporary.CustomName = UsedRegisters.at(Register).CustomName;
        Result.emplace_back(Temporary);
      } else if (MustUseTheNextOne) {
        if constexpr (!AT::OnlyStartDoubleArgumentsFromAnEvenRegister) {
          return std::nullopt;
        } else if ((Index & 1) == 0) {
          return std::nullopt;
        } else if (Result.size() > 1 && Index > 1) {
          auto &First = Result[Result.size() - 1];
          auto &Second = Result[Result.size() - 2];
          if (!First.CustomName.empty() || !Second.CustomName.empty()) {
            if (First.CustomName.empty())
              First.CustomName = "unnamed";
            if (Second.CustomName.empty())
              Second.CustomName = "unnamed";
            First.CustomName.append(("+" + Second.CustomName).str());
          }
          if constexpr (!DryRun) {
            auto NewType = buildDoubleType(AllowedRegisters.at(Index - 2),
                                           AllowedRegisters.at(Index - 1),
                                           model::PrimitiveTypeKind::Generic,
                                           TheBinary);
            if (NewType == std::nullopt)
              return std::nullopt;

            First.Type = *NewType;
          }
          Result.pop_back();
        } else {
          return std::nullopt;
        }
      }

      MustUseTheNextOne = MustUseTheNextOne || IsUsed;
    }

    for (auto Pair : llvm::enumerate(llvm::reverse(Result)))
      Pair.value().Index = Pair.index();

    return Result;
  }

  template<typename RegisterType, size_t RegisterCount, bool DryRun = false>
  static std::optional<model::QualifiedType>
  convertReturnValue(const SortedVector<RegisterType> &UsedRegisters,
                     const RegisterArray<RegisterCount> &AllowedRegisters,
                     const model::Register::Values PointerToCopyLocation,
                     model::Binary &TheBinary) {
    if (UsedRegisters.size() == 0) {
      auto Void = TheBinary.getPrimitiveType(model::PrimitiveTypeKind::Void, 0);
      return model::QualifiedType{ Void, {} };
    }

    if (UsedRegisters.size() == 1) {
      if (UsedRegisters.begin()->Location == PointerToCopyLocation) {
        if constexpr (DryRun)
          return model::QualifiedType{};
        else
          return getTypeOrDefault(UsedRegisters.begin()->Type,
                                  PointerToCopyLocation,
                                  TheBinary);
      } else {
        if constexpr (RegisterCount == 0)
          return std::nullopt;
        if (AllowedRegisters.front() == UsedRegisters.begin()->Location) {
          if constexpr (DryRun)
            return model::QualifiedType{};
          else
            return getTypeOrDefault(UsedRegisters.begin()->Type,
                                    UsedRegisters.begin()->Location,
                                    TheBinary);
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
          CurrentField.Offset = ReturnStruct->Size;
          if constexpr (!DryRun)
            CurrentField.Type = getTypeOrDefault(UsedIterator->Type,
                                                 UsedIterator->Location,
                                                 TheBinary);
          ReturnStruct->Fields.insert(std::move(CurrentField));

          ReturnStruct->Size += model::Register::getSize(Register);
        } else if (MustUseTheNextOne) {
          if constexpr (!AT::OnlyStartDoubleArgumentsFromAnEvenRegister)
            return std::nullopt;
          else if ((Index & 1) == 0 || ReturnStruct->Fields.size() <= 1
                   || Index <= 1)
            return std::nullopt;
        }

        MustUseTheNextOne = MustUseTheNextOne || IsCurrentRegisterUsed;
      }

      revng_assert(ReturnStruct->Size != 0 && !ReturnStruct->Fields.empty());

      if constexpr (DryRun) {
        auto ReturnStructTypePath = TheBinary.recordNewType(std::move(Result));
        revng_assert(ReturnStructTypePath.isValid());
        return model::QualifiedType{ ReturnStructTypePath, {} };
      } else {
        return model::QualifiedType{};
      }
    }

    return std::nullopt;
  }

  template<typename RType, size_t RCount>
  static bool verifyArgumentsToBeConvertible(const SortedVector<RType> &UR,
                                             const RegisterArray<RCount> &AR,
                                             model::Binary &B) {
    return convertArguments<RType, RCount, true>(UR, AR, B).has_value();
  }

  template<typename RType, size_t RCount>
  static bool
  verifyReturnValueToBeConvertible(const SortedVector<RType> &UR,
                                   const RegisterArray<RCount> &AR,
                                   const model::Register::Values PtC,
                                   model::Binary &B) {
    return convertReturnValue<RType, RCount, true>(UR, AR, PtC, B).has_value();
  }

  static DistributedArguments
  distributePositionBasedArguments(const ArgumentContainer &Arguments) {
    DistributedArguments Result;

    for (const model::Argument &Argument : Arguments) {
      if (Result.size() <= Argument.Index)
        Result.resize(Argument.Index + 1);
      auto &Distributed = Result[Argument.Index];

      auto MaybeSize = Argument.Type.size();
      revng_assert(MaybeSize.has_value());
      Distributed.Size = *MaybeSize;

      if (Argument.Type.isFloat()) {
        if (Argument.Index < AT::VectorArgumentRegisters.size()) {
          auto Register = AT::VectorArgumentRegisters[Argument.Index];
          Distributed.Registers.emplace_back(Register);
        } else {
          Distributed.SizeOnStack = Distributed.Size;
        }
      } else {
        if (Argument.Index < AT::GeneralPurposeArgumentRegisters.size()) {
          auto Reg = AT::GeneralPurposeArgumentRegisters[Argument.Index];
          Distributed.Registers.emplace_back(Reg);
        } else {
          Distributed.SizeOnStack = Distributed.Size;
        }
      }
    }

    return Result;
  }

  static constexpr auto UnlimitedRegisters = std::numeric_limits<size_t>::max();

  template<size_t RegisterCount>
  static std::pair<DistributedArgument, size_t>
  considerRegisters(size_t Size,
                    size_t AllowedRegisterLimit,
                    size_t OccupiedRegisterCount,
                    const RegisterArray<RegisterCount> &AllowedRegisters,
                    bool AllowPuttingPartOfAnArgumentOnStack) {
    size_t RegisterLimit = OccupiedRegisterCount + AllowedRegisterLimit;
    size_t ConsideredRegisterCounter = OccupiedRegisterCount;

    size_t SizeCounter = 0;
    const size_t ARC = AllowedRegisters.size();
    if (ARC > 0) {
      size_t &CRC = ConsideredRegisterCounter;
      while (SizeCounter < Size && CRC < ARC && CRC < RegisterLimit) {
        size_t RegisterIndex = ConsideredRegisterCounter++;
        auto CurrentRegister = AllowedRegisters[RegisterIndex];
        SizeCounter += model::Register::getSize(CurrentRegister);
      }
    }

    DistributedArgument DA;
    DA.Size = Size;

    if constexpr (AT::OnlyStartDoubleArgumentsFromAnEvenRegister) {
      if (ConsideredRegisterCounter - OccupiedRegisterCount == 2) {
        if ((OccupiedRegisterCount & 1) != 0) {
          ++OccupiedRegisterCount;
          ++ConsideredRegisterCounter;
        }
      }
    }

    if (SizeCounter >= Size) {
      for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
        DA.Registers.emplace_back(AllowedRegisters[I]);
      DA.SizeOnStack = 0;
    } else if (AllowPuttingPartOfAnArgumentOnStack) {
      for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
        DA.Registers.emplace_back(AllowedRegisters[I]);
      DA.SizeOnStack = DA.Size - SizeCounter;
    } else {
      DA.SizeOnStack = DA.Size;
      ConsideredRegisterCounter = OccupiedRegisterCount;
    }

    return { DA, ConsideredRegisterCounter };
  }

  static DistributedArguments
  distributeNonPositionBasedArguments(const ArgumentContainer &Arguments) {
    DistributedArguments Result;
    size_t UsedGeneralPurposeRegisterCounter = 0;
    size_t UsedVectorRegisterCounter = 0;

    for (const model::Argument &Argument : Arguments) {
      auto MaybeSize = Argument.Type.size();
      revng_assert(MaybeSize.has_value());

      constexpr bool CanSplit = AT::ArgumentsCanBeSplitBetweenRegistersAndStack;
      if (Argument.Type.isFloat()) {
        static constexpr auto &Registers = AT::VectorArgumentRegisters;
        size_t &Counter = UsedVectorRegisterCounter;
        const size_t Limit = 1;

        auto [Distributed, NextIndex] = considerRegisters(*MaybeSize,
                                                          Limit,
                                                          Counter,
                                                          Registers,
                                                          CanSplit);
        if (Result.size() <= Argument.Index)
          Result.resize(Argument.Index + 1);
        Result[Argument.Index] = Distributed;
        Counter = NextIndex;
      } else {
        static constexpr auto &Registers = AT::GeneralPurposeArgumentRegisters;
        size_t &Counter = UsedGeneralPurposeRegisterCounter;
        if (Argument.Type.isScalar()) {
          const size_t Limit = AT::MaximumGPRsPerScalarArgument;

          auto [Distributed, NextIndex] = considerRegisters(*MaybeSize,
                                                            Limit,
                                                            Counter,
                                                            Registers,
                                                            CanSplit);
          if (Result.size() <= Argument.Index)
            Result.resize(Argument.Index + 1);
          Result[Argument.Index] = Distributed;
          Counter = NextIndex;
        } else {
          const size_t Limit = AT::MaximumGPRsPerAggregateArgument;

          auto [Distributed, NextIndex] = considerRegisters(*MaybeSize,
                                                            Limit,
                                                            Counter,
                                                            Registers,
                                                            CanSplit);
          if (Result.size() <= Argument.Index)
            Result.resize(Argument.Index + 1);
          Result[Argument.Index] = Distributed;
          Counter = NextIndex;
        }
      }
    }

    return Result;
  }

public:
  static DistributedArguments
  distributeArguments(const ArgumentContainer &Arguments) {
    if constexpr (AT::ArgumentsArePositionBased)
      return distributePositionBasedArguments(Arguments);
    else
      return distributeNonPositionBasedArguments(Arguments);
  }

  static DistributedArgument
  distributeReturnValue(const model::QualifiedType &ReturnValueType) {
    if (ReturnValueType.isVoid())
      return DistributedArgument{};

    auto MaybeSize = ReturnValueType.size();
    revng_assert(MaybeSize.has_value());

    if (ReturnValueType.isFloat()) {
      const auto &Registers = AT::VectorReturnValueRegisters;
      // TODO: replace `UnlimitedRegisters` with the actual value to be defined
      //       by the trait.
      return considerRegisters(*MaybeSize,
                               UnlimitedRegisters,
                               0,
                               Registers,
                               false)
        .first;
    } else {
      const auto &Registers = AT::GeneralPurposeReturnValueRegisters;
      if (ReturnValueType.isScalar()) {
        const size_t L = AT::MaximumGPRsPerScalarReturnValue;
        return considerRegisters(*MaybeSize, L, 0, Registers, false).first;
      } else {
        const size_t L = AT::MaximumGPRsPerAggregateReturnValue;
        return considerRegisters(*MaybeSize, L, 0, Registers, false).first;
      }
    }
  }

private:
  static model::QualifiedType
  chooseArgumentType(const model::QualifiedType &ArgumentType,
                     model::Register::Values Register,
                     const RegisterList &RegisterList,
                     model::Binary &TheBinary) {
    if (RegisterList.size() > 1) {
      return buildGenericType(Register, TheBinary);
    } else {
      auto ResultType = ArgumentType;
      auto MaybeSize = ArgumentType.size();
      auto TargetSize = model::Register::getSize(Register);

      if (!MaybeSize.has_value()) {
        return buildType(Register, TheBinary);
      } else if (*MaybeSize > TargetSize) {
        auto Qualifier = model::Qualifier::createPointer(TargetSize);
        ResultType.Qualifiers.emplace_back(Qualifier);
      } else if (!ResultType.isScalar()) {
        return buildGenericType(Register, TheBinary);
      }

      return ResultType;
    }
  }
};

std::optional<model::CABIFunctionType>
tryConvertToCABI(const model::RawFunctionType &Function,
                 model::Binary &TheBinary,
                 std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = TheBinary.DefaultABI;
  revng_assert(*MaybeABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(*MaybeABI, [&]<model::ABI::Values A>() {
    return ConversionHelper<A>::toCABI(Function, TheBinary);
  });
}

model::RawFunctionType convertToRaw(const model::CABIFunctionType &Function,
                                    model::Binary &TheBinary) {
  revng_assert(Function.ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(Function.ABI, [&]<model::ABI::Values A>() {
    return ConversionHelper<A>::toRaw(Function, TheBinary);
  });
}

Layout::Layout(const model::CABIFunctionType &Function) :
  Layout(skippingEnumSwitch<1>(Function.ABI, [&]<model::ABI::Values A>() {
    Layout Result;

    size_t CurrentOffset = 0;
    auto Args = ConversionHelper<A>::distributeArguments(Function.Arguments);
    revng_assert(Args.size() == Function.Arguments.size());
    for (size_t Index = 0; Index < Args.size(); ++Index) {
      auto &Current = Result.Arguments.emplace_back();
      Current.Registers = std::move(Args[Index].Registers);
      if (Args[Index].SizeOnStack != 0) {
        // TODO: maybe some kind of alignment considerations are needed here.
        Current.Stack = typename Layout::Argument::StackSpan{
          CurrentOffset, Args[Index].SizeOnStack
        };
        CurrentOffset += Args[Index].SizeOnStack;
      }
    }

    auto RV = ConversionHelper<A>::distributeReturnValue(Function.ReturnType);
    revng_assert(RV.SizeOnStack == 0);
    Result.ReturnValue.Registers = std::move(RV.Registers);

    using AT = abi::Trait<A>;
    Result.CalleeSavedRegisters.resize(AT::CalleeSavedRegisters.size());
    llvm::copy(AT::CalleeSavedRegisters, Result.CalleeSavedRegisters.begin());

    Result.FinalStackOffset = 0; // TODO: calculate this offset properly.

    return Result;
  })) {
}

Layout::Layout(const model::RawFunctionType &Function) {
  // Lay register arguments out.
  for (const model::NamedTypedRegister &Register : Function.Arguments)
    Arguments.emplace_back().Registers = { Register.Location };

  // Lay stack arguments out.
  if (Function.StackArgumentsType.isValid()) {
    const model::Type *OriginalStackType = Function.StackArgumentsType.get();
    auto *StackStruct = llvm::dyn_cast<model::StructType>(OriginalStackType);
    revng_assert(StackStruct,
                 "`RawFunctionType::StackArgumentsType` must be a struct.");
    typename Layout::Argument::StackSpan StackSpan{ 0, StackStruct->Size };
    Arguments.emplace_back().Stack = std::move(StackSpan);
  }

  // Lay the return value out.
  for (const model::TypedRegister &Register : Function.ReturnValues)
    ReturnValue.Registers.emplace_back(Register.Location);

  // Fill callee saved registers.
  CalleeSavedRegisters.resize(Function.PreservedRegisters.size());
  llvm::copy(Function.PreservedRegisters, CalleeSavedRegisters.begin());
}

bool Layout::verify() const {
  model::Architecture::Values ExpectedArch = model::Architecture::Invalid;
  std::unordered_set<model::Register::Values> LookupHelper;
  auto VerificationHelper = [&](model::Register::Values Register) -> bool {
    // Ensure each register is present only once
    if (!LookupHelper.emplace(Register).second)
      return false;

    // Ensure all the registers belong to the same architecture
    if (ExpectedArch == model::Architecture::Invalid)
      ExpectedArch = model::Register::getArchitecture(Register);
    else if (ExpectedArch != model::Register::getArchitecture(Register))
      return false;

    return true;
  };

  // Verify arguments
  LookupHelper.clear();
  for (const Layout::Argument &Argument : Arguments)
    for (model::Register::Values Register : Argument.Registers)
      if (!VerificationHelper(Register))
        return false;

  // Verify return values
  LookupHelper.clear();
  for (model::Register::Values Register : ReturnValue.Registers)
    if (!VerificationHelper(Register))
      return false;

  return true;
}

size_t Layout::argumentRegisterCount() const {
  size_t Result = 0;

  for (auto &Argument : Arguments)
    Result += Argument.Registers.size();

  return Result;
}

size_t Layout::returnValueRegisterCount() const {
  return ReturnValue.Registers.size();
}

} // namespace abi::FunctionType
