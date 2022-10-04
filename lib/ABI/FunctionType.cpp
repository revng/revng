/// \file FunctionType.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "revng/ABI/FunctionType.h"
#include "revng/ABI/RegisterOrder.h"
#include "revng/ABI/RegisterStateDeductions.h"
#include "revng/ABI/Trait.h"
#include "revng/ADT/STLExtras.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Generated/Early/TypeKind.h"
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

constexpr static model::PrimitiveTypeKind::Values
selectTypeKind(model::Register::Values) {
  // TODO: implement a way to determine the register type. At the very least
  // we should be able to differentiate GPRs from the vector registers.

  return model::PrimitiveTypeKind::PointerOrNumber;
}

static model::QualifiedType
buildType(model::Register::Values Register, model::Binary &Binary) {
  model::PrimitiveTypeKind::Values Kind = selectTypeKind(Register);
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(Binary.getPrimitiveType(Kind, Size), {});
}

static model::QualifiedType
buildGenericType(model::Register::Values Register, model::Binary &Binary) {
  constexpr auto Kind = model::PrimitiveTypeKind::Generic;
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(Binary.getPrimitiveType(Kind, Size), {});
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

static void replaceReferences(const model::Type::Key &OldKey,
                              const model::TypePath &NewTypePath,
                              TupleTree<model::Binary> &Model) {
  auto Visitor = [&](model::TypePath &Visited) {
    if (!Visited.isValid())
      return; // Ignore empty references

    model::Type *Current = Visited.get();
    revng_assert(Current != nullptr);

    if (Current->key() == OldKey)
      Visited = NewTypePath;
  };
  Model.visitReferences(Visitor);
  Model->Types().erase(OldKey);
}

/// A helper used to differentiate vector registers.
///
/// \param Register Any CPU register the model is aware of.
///
/// \return `true` if `Register` is a vector register, `false` otherwise.
constexpr bool isVectorRegister(model::Register::Values Register) {
  namespace PrimitiveTypeKind = model::PrimitiveTypeKind;
  return model::Register::primitiveKind(Register) == PrimitiveTypeKind::Float;
}

/// Helps detecting unsupported ABI trait definition with respect to
/// the way they return the return values.
///
/// This is an important piece of abi trait verification. For more information
/// see the `static_assert` that invokes it in \ref distributeArguments
///
/// \note Make this function `consteval` after clang-14 is available.
///
/// \tparam ABI the ABI, `abi::Trait` of which is to be checked.
///
/// \return `true` if the ABI is valid, `false` otherwise.
template<model::ABI::Values ABI>
constexpr bool verifyReturnValueLocationRegister() {
  using AT = abi::Trait<ABI>;
  constexpr auto &LocationRegister = AT::ReturnValueLocationRegister;

  // Skip ABIs that do not allow returning big values.
  // They do not benefit from this check.
  if constexpr (LocationRegister != model::Register::Invalid) {
    if constexpr (isVectorRegister(LocationRegister)) {
      // Vector register used as the return value locations are not supported.
      return false;
    } else if constexpr (revng::is_contained(AT::CalleeSavedRegisters,
                                             LocationRegister)) {
      // Using callee saved register as a return value location doesn't make
      // much sense: filter those out.
      return false;
    } else {
      // The return value location register can optionally also be the first
      // GPRs, but only the first one.
      constexpr auto &GPRs = AT::GeneralPurposeArgumentRegisters;
      constexpr auto Iterator = revng::find(GPRs, LocationRegister);
      if constexpr (Iterator != GPRs.end() && Iterator != GPRs.begin())
        return false;
    }
  }

  return true;
}

namespace ModelArch = model::Architecture;

template<model::ABI::Values ABI>
class ConversionHelper {
  using AT = abi::Trait<ABI>;
  static constexpr auto Architecture = model::ABI::getArchitecture(ABI);
  static constexpr auto RegisterSize = ModelArch::getPointerSize(Architecture);

  using IndexType = decay_t<model::Argument::IndexType>;
  using RegisterList = llvm::SmallVector<model::Register::Values, 1>;
  struct DistributedArgument {
    RegisterList Registers = {};
    size_t Size = 0, SizeOnStack = 0;
  };
  using DistributedArguments = llvm::SmallVector<DistributedArgument, 4>;

  using ArgumentContainer = SortedVector<model::Argument>;

public:
  static std::optional<model::TypePath>
  toCABI(const model::RawFunctionType &Function,
         TupleTree<model::Binary> &Binary) {
    static constexpr auto Arch = model::ABI::getArchitecture(ABI);
    if (!verify<Arch>(Function.Arguments(),
                      AT::GeneralPurposeArgumentRegisters))
      return std::nullopt;
    if (!verify<Arch>(Function.ReturnValues(),
                      AT::GeneralPurposeReturnValueRegisters))
      return std::nullopt;

    // Verify the architecture of return value location register if present.
    constexpr model::Register::Values PTCRR = AT::ReturnValueLocationRegister;
    if (PTCRR != model::Register::Invalid)
      revng_assert(model::Register::getReferenceArchitecture(PTCRR) == Arch);

    // Verify the architecture of callee saved registers.
    for (auto &Register : AT::CalleeSavedRegisters)
      revng_assert(model::Register::getReferenceArchitecture(Register) == Arch);

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

    auto StackArgumentList = convertStackArguments(Function
                                                     .StackArgumentsType(),
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

    // Replace all references to the old type with references to the new one.
    replaceReferences(Function.key(), NewTypePath, Binary);

    return NewTypePath;
  }

  static model::TypePath toRaw(const model::CABIFunctionType &Function,
                               TupleTree<model::Binary> &Binary) {
    // TODO: fix the return value distribution.
    auto Arguments = distributeArguments(Function.Arguments(), 0);

    model::RawFunctionType Result;
    Result.CustomName() = Function.CustomName();
    Result.OriginalName() = Function.OriginalName();

    model::StructType StackArguments;
    uint64_t CombinedStackArgumentSize = 0;
    for (size_t ArgIndex = 0; ArgIndex < Arguments.size(); ++ArgIndex) {
      auto &ArgumentStorage = Arguments[ArgIndex];
      const auto &ArgumentType = Function.Arguments().at(ArgIndex).Type();
      if (!ArgumentStorage.Registers.empty()) {
        // Handle the registers
        auto ArgumentName = Function.Arguments().at(ArgIndex).name();
        for (size_t Index = 0; auto Register : ArgumentStorage.Registers) {
          model::NamedTypedRegister Argument(Register);
          Argument.Type() = chooseArgumentType(ArgumentType,
                                               Register,
                                               ArgumentStorage.Registers,
                                               *Binary);

          // TODO: see what can be done to preserve names better
          if (llvm::StringRef{ ArgumentName.str() }.take_front(8) != "unnamed_")
            Argument.CustomName() = ArgumentName;

          Result.Arguments().insert(Argument);
        }
      }

      if (ArgumentStorage.SizeOnStack != 0) {
        // Handle the stack
        auto ArgumentIterator = Function.Arguments().find(ArgIndex);
        revng_assert(ArgumentIterator != Function.Arguments().end());
        const model::Argument &Argument = *ArgumentIterator;

        model::StructField Field;
        Field.Offset() = CombinedStackArgumentSize;
        Field.CustomName() = Argument.CustomName();
        Field.OriginalName() = Argument.OriginalName();
        Field.Type() = Argument.Type();
        StackArguments.Fields().insert(std::move(Field));

        // Compute the full size of the argument (including padding if needed).
        auto MaybeSize = Argument.Type().size();
        revng_assert(MaybeSize.has_value() && MaybeSize.value() != 0);
        CombinedStackArgumentSize += paddedSizeOnStack(MaybeSize.value());
      }
    }

    if (CombinedStackArgumentSize != 0) {
      StackArguments.Size() = CombinedStackArgumentSize;

      using namespace model;
      auto Type = UpcastableType::make<StructType>(std::move(StackArguments));
      Result.StackArgumentsType() = { Binary->recordNewType(std::move(Type)),
                                      {} };
    }

    Result.FinalStackOffset() = finalStackOffset(Arguments);

    if (!Function.ReturnType().isVoid()) {
      auto ReturnValue = distributeReturnValue(Function.ReturnType());
      if (!ReturnValue.Registers.empty()) {
        // Handle a register-based return value.
        for (model::Register::Values Register : ReturnValue.Registers) {
          model::TypedRegister ReturnValueRegister;
          ReturnValueRegister.Location() = Register;
          ReturnValueRegister.Type() = chooseArgumentType(Function.ReturnType(),
                                                          Register,
                                                          ReturnValue.Registers,
                                                          *Binary);

          Result.ReturnValues().insert(std::move(ReturnValueRegister));
        }

        // Try and recover types from the struct if possible
        if (Function.ReturnType().Qualifiers().empty()) {
          const model::Type
            *Type = Function.ReturnType().UnqualifiedType().get();
          revng_assert(Type != nullptr);
          const auto *Struct = llvm::dyn_cast<model::StructType>(Type);
          if (Struct
              && Struct->Fields().size() == Result.ReturnValues().size()) {
            using RegisterEnum = model::Register::Values;
            SmallMap<RegisterEnum, model::QualifiedType, 4> RecoveredTypes;
            size_t StructOffset = 0;
            for (size_t Index = 0; Index < Struct->Fields().size(); ++Index) {
              if (Index >= AT::GeneralPurposeReturnValueRegisters.size())
                break;
              auto Register = AT::GeneralPurposeReturnValueRegisters[Index];

              auto TypedRegisterIterator = Result.ReturnValues().find(Register);
              if (TypedRegisterIterator == Result.ReturnValues().end())
                break;

              const auto &Field = Struct->Fields().at(StructOffset);

              auto MaybeFieldSize = Field.Type().size();
              revng_assert(MaybeFieldSize != std::nullopt);

              auto MaybeRegisterSize = TypedRegisterIterator->Type().size();
              revng_assert(MaybeRegisterSize != std::nullopt);

              if (MaybeFieldSize.value() != MaybeRegisterSize.value())
                break;

              auto Tie = std::tie(Register, Field.Type());
              auto [Iterator, Success] = RecoveredTypes.insert(std::move(Tie));
              revng_assert(Success);

              StructOffset += MaybeFieldSize.value();
            }

            if (RecoveredTypes.size() == Result.ReturnValues().size())
              for (auto [Register, Type] : RecoveredTypes)
                Result.ReturnValues().at(Register).Type() = Type;
          }
        }
      } else {
        // Handle a pointer-based return value.
        revng_assert(!AT::GeneralPurposeReturnValueRegisters.empty());
        auto Register = AT::GeneralPurposeReturnValueRegisters[0];
        auto RegisterSize = model::Register::getSize(Register);
        auto PointerQualifier = model::Qualifier::createPointer(RegisterSize);

        auto MaybeReturnValueSize = Function.ReturnType().size();
        revng_assert(MaybeReturnValueSize != std::nullopt);
        revng_assert(ReturnValue.Size == *MaybeReturnValueSize);

        model::QualifiedType ReturnType = Function.ReturnType();
        ReturnType.Qualifiers().emplace_back(PointerQualifier);

        model::TypedRegister ReturnPointer(Register);
        ReturnPointer.Type() = std::move(ReturnType);
        Result.ReturnValues().insert(std::move(ReturnPointer));
      }
    }

    // Populate the list of preserved registers
    for (model::Register::Values Register : AT::CalleeSavedRegisters)
      Result.PreservedRegisters().insert(Register);

    // Steal the ID
    Result.ID() = Function.ID();

    // Add converted type to the model.
    using UT = model::UpcastableType;
    auto Ptr = UT::make<model::RawFunctionType>(std::move(Result));
    auto NewTypePath = Binary->recordNewType(std::move(Ptr));

    // Replace all references to the old type with references to the new one.
    replaceReferences(Function.key(), NewTypePath, Binary);

    return NewTypePath;
  }

  static uint64_t finalStackOffset(const DistributedArguments &Arguments) {
    constexpr auto Architecture = model::ABI::getArchitecture(ABI);
    uint64_t Result = model::Architecture::getCallPushSize(Architecture);

    if constexpr (AT::CalleeIsResponsibleForStackCleanup) {
      for (auto &Argument : Arguments)
        Result += Argument.SizeOnStack;

      // TODO: take return values into the account.

      // TODO: take shadow space into the account if relevant.

      static_assert((AT::StackAlignment & (AT::StackAlignment - 1)) == 0);
      Result += AT::StackAlignment - 1;
      Result &= ~(AT::StackAlignment - 1);
    }

    return Result;
  }

private:
  /// Takes care of extending (padding) the size of a stack argument.
  ///
  /// \note This only accounts for the post-padding (extension).
  ///       Pre-padding (offset) needs to be taken care of separately.
  ///
  /// \param RealSize The size of the argument without the padding.
  ///
  /// \return The size of the argument with the padding.
  static uint64_t paddedSizeOnStack(uint64_t RealSize) {
    if (RealSize <= RegisterSize) {
      RealSize = RegisterSize;
    } else {
      static_assert((RegisterSize & (RegisterSize - 1)) == 0);
      RealSize += RegisterSize - 1;
      RealSize &= ~(RegisterSize - 1);
    }

    return RealSize;
  }

  template<typename RegisterType, size_t RegisterCount, bool DryRun = false>
  static std::optional<llvm::SmallVector<model::Argument, 8>>
  convertArguments(const SortedVector<RegisterType> &UsedRegisters,
                   const RegisterArray<RegisterCount> &AllowedRegisters,
                   model::Binary &Binary) {
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

  static llvm::SmallVector<model::Argument, 8>
  convertStackArguments(model::QualifiedType StackArgumentTypes,
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

  template<typename RegisterType, size_t RegisterCount, bool DryRun = false>
  static std::optional<model::QualifiedType>
  convertReturnValue(const SortedVector<RegisterType> &UsedRegisters,
                     const RegisterArray<RegisterCount> &AllowedRegisters,
                     const model::Register::Values PointerToCopyLocation,
                     model::Binary &Binary) {
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
        if constexpr (RegisterCount == 0)
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

      revng_assert(ReturnStruct->Size() != 0
                   && !ReturnStruct->Fields().empty());

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
  distributePositionBasedArguments(const ArgumentContainer &Arguments,
                                   std::size_t SkippedRegisters = 0) {
    DistributedArguments Result;

    for (const model::Argument &Argument : Arguments) {
      std::size_t RegisterIndex = Argument.Index() + SkippedRegisters;
      if (Result.size() <= RegisterIndex)
        Result.resize(RegisterIndex + 1);
      auto &Distributed = Result[RegisterIndex];

      auto MaybeSize = Argument.Type().size();
      revng_assert(MaybeSize.has_value());
      Distributed.Size = *MaybeSize;

      if (Argument.Type().isFloat()) {
        if (RegisterIndex < AT::VectorArgumentRegisters.size()) {
          auto Register = AT::VectorArgumentRegisters[RegisterIndex];
          Distributed.Registers.emplace_back(Register);
        } else {
          Distributed.SizeOnStack = paddedSizeOnStack(Distributed.Size);
        }
      } else {
        if (RegisterIndex < AT::GeneralPurposeArgumentRegisters.size()) {
          auto Reg = AT::GeneralPurposeArgumentRegisters[RegisterIndex];
          Distributed.Registers.emplace_back(Reg);
        } else {
          Distributed.SizeOnStack = paddedSizeOnStack(Distributed.Size);
        }
      }
    }

    return Result;
  }

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

    if (DA.SizeOnStack != 0)
      DA.SizeOnStack = paddedSizeOnStack(DA.SizeOnStack);

    return { DA, ConsideredRegisterCounter };
  }

  static DistributedArguments
  distributeNonPositionBasedArguments(const ArgumentContainer &Arguments,
                                      std::size_t SkippedRegisters = 0) {
    DistributedArguments Result;
    size_t UsedGeneralPurposeRegisterCount = SkippedRegisters;
    size_t UsedVectorRegisterCount = 0;

    for (const model::Argument &Argument : Arguments) {
      auto MaybeSize = Argument.Type().size();
      revng_assert(MaybeSize.has_value());

      constexpr bool CanSplit = AT::ArgumentsCanBeSplitBetweenRegistersAndStack;
      if (Argument.Type().isFloat()) {
        // The conventional non-position based approach is not applicable for
        // vector registers since it's rare for multiple registers to be used
        // to pass a single argument.
        //
        // For now, just return the next free register.
        //
        // TODO: handle this properly.

        static constexpr auto &Registers = AT::VectorArgumentRegisters;
        if (UsedVectorRegisterCount < Registers.size()) {
          // There is a free register to put the argument in.
          if (Result.size() <= Argument.Index())
            Result.resize(Argument.Index() + 1);

          auto Register = Registers[UsedVectorRegisterCount];
          Result[Argument.Index()].Registers.emplace_back(Register);
          Result[Argument.Index()].Size = *MaybeSize;
          Result[Argument.Index()].SizeOnStack = 0;

          UsedVectorRegisterCount++;
        } else {
          // There are no more free registers left,
          // pass the argument on the stack.
          if (Result.size() <= Argument.Index())
            Result.resize(Argument.Index() + 1);
          Result[Argument.Index()].Size = *MaybeSize;
          Result[Argument.Index()].SizeOnStack = paddedSizeOnStack(*MaybeSize);
        }
      } else {
        static constexpr auto &Registers = AT::GeneralPurposeArgumentRegisters;
        size_t &Counter = UsedGeneralPurposeRegisterCount;
        if (Argument.Type().isScalar()) {
          const size_t Limit = AT::MaximumGPRsPerScalarArgument;

          auto [Distributed, NextIndex] = considerRegisters(*MaybeSize,
                                                            Limit,
                                                            Counter,
                                                            Registers,
                                                            CanSplit);
          if (Result.size() <= Argument.Index())
            Result.resize(Argument.Index() + 1);
          Result[Argument.Index()] = Distributed;
          Counter = NextIndex;
        } else {
          const size_t Limit = AT::MaximumGPRsPerAggregateArgument;

          auto [Distributed, NextIndex] = considerRegisters(*MaybeSize,
                                                            Limit,
                                                            Counter,
                                                            Registers,
                                                            CanSplit);
          if (Result.size() <= Argument.Index())
            Result.resize(Argument.Index() + 1);
          Result[Argument.Index()] = Distributed;
          Counter = NextIndex;
        }
      }
    }

    return Result;
  }

public:
  static DistributedArguments
  distributeArguments(const ArgumentContainer &Arguments,
                      bool PassesReturnValueLocationAsAnArgument) {
    bool SkippedRegisters = 0;
    if (PassesReturnValueLocationAsAnArgument == true) {
      static_assert(verifyReturnValueLocationRegister<ABI>(),
                    "ABIs where non-first argument GPR is used for passing the "
                    "address of the space allocated for big return values are "
                    "not currently supported.");

      static constexpr auto &GPRs = AT::GeneralPurposeArgumentRegisters;
      if constexpr (!GPRs.empty())
        if (AT::ReturnValueLocationRegister == GPRs[0])
          SkippedRegisters = 1;
    }

    if constexpr (AT::ArgumentsArePositionBased)
      return distributePositionBasedArguments(Arguments, SkippedRegisters);
    else
      return distributeNonPositionBasedArguments(Arguments, SkippedRegisters);
  }

  static DistributedArgument
  distributeReturnValue(const model::QualifiedType &ReturnValueType) {
    if (ReturnValueType.isVoid())
      return DistributedArgument{};

    auto MaybeSize = ReturnValueType.size();
    revng_assert(MaybeSize.has_value());

    if (ReturnValueType.isFloat()) {
      const auto &Registers = AT::VectorReturnValueRegisters;

      // As a temporary measure always just return the first vector return value
      // register, if it's available.
      //
      // TODO: handle the situation properly.
      if constexpr (!AT::VectorReturnValueRegisters.empty()) {
        DistributedArgument Result;
        Result.Size = *MaybeSize;
        Result.Registers.emplace_back(Registers.front());
        Result.SizeOnStack = 0;
        return Result;
      } else {
        // If there are no vector registers, dedicated to be used for passing
        // arguments in the current ABI, use stack instead.
        return considerRegisters(*MaybeSize, 0, 0, Registers, false).first;
      }
    } else {
      const size_t L = ReturnValueType.isScalar() ?
                         AT::MaximumGPRsPerScalarReturnValue :
                         AT::MaximumGPRsPerAggregateReturnValue;
      constexpr auto &Registers = AT::GeneralPurposeReturnValueRegisters;
      return considerRegisters(*MaybeSize, L, 0, Registers, false).first;
    }
  }

private:
  static model::QualifiedType
  chooseArgumentType(const model::QualifiedType &ArgumentType,
                     model::Register::Values Register,
                     const RegisterList &RegisterList,
                     model::Binary &Binary) {
    if (RegisterList.size() > 1) {
      return buildGenericType(Register, Binary);
    } else {
      auto ResultType = ArgumentType;
      auto MaybeSize = ArgumentType.size();
      auto TargetSize = model::Register::getSize(Register);

      if (!MaybeSize.has_value()) {
        return buildType(Register, Binary);
      } else if (*MaybeSize > TargetSize) {
        auto Qualifier = model::Qualifier::createPointer(TargetSize);
        ResultType.Qualifiers().emplace_back(Qualifier);
      } else if (!ResultType.isScalar()) {
        return buildGenericType(Register, Binary);
      }

      return ResultType;
    }
  }
};

std::optional<model::TypePath>
tryConvertToCABI(const model::RawFunctionType &Function,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> MaybeABI) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();
  revng_assert(*MaybeABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(*MaybeABI, [&]<model::ABI::Values A>() {
    return ConversionHelper<A>::toCABI(Function, Binary);
  });
}

model::TypePath convertToRaw(const model::CABIFunctionType &Function,
                             TupleTree<model::Binary> &Binary) {
  revng_assert(Function.ABI() != model::ABI::Invalid);
  return skippingEnumSwitch<1>(Function.ABI(), [&]<model::ABI::Values A>() {
    return ConversionHelper<A>::toRaw(Function, Binary);
  });
}

Layout::Layout(const model::CABIFunctionType &Function) :
  Layout(skippingEnumSwitch<1>(Function.ABI(), [&]<model::ABI::Values A>() {
    Layout Result;

    using AT = abi::Trait<A>;
    static constexpr auto Arch = model::ABI::getArchitecture(A);
    auto RV = ConversionHelper<A>::distributeReturnValue(Function.ReturnType());
    if (RV.SizeOnStack == 0) {
      // Nothing on the stack, the return value fits into the registers.
      auto &ReturnValue = Result.ReturnValues.emplace_back();
      ReturnValue.Type = Function.ReturnType();
      ReturnValue.Registers = std::move(RV.Registers);
    } else {
      revng_assert(RV.Registers.empty(),
                   "Register and stack return values should never be present "
                   "at the same time.");
      revng_assert(AT::ReturnValueLocationRegister != model::Register::Invalid,
                   "Big return values are not supported by the current ABI");
      auto &RVLocationArg = Result.Arguments.emplace_back();
      RVLocationArg.Registers.emplace_back(AT::ReturnValueLocationRegister);
      RVLocationArg.Type = Function.ReturnType().getPointerTo(Arch);
      RVLocationArg.Kind = ArgumentKind::ShadowPointerToAggregateReturnValue;
    }

    size_t CurrentOffset = 0;
    auto Args = ConversionHelper<A>::distributeArguments(Function.Arguments(),
                                                         RV.SizeOnStack != 0);
    revng_assert(Args.size() == Function.Arguments().size());
    for (size_t Index = 0; Index < Args.size(); ++Index) {
      auto &Current = Result.Arguments.emplace_back();
      const model::QualifiedType
        &ArgumentType = Function.Arguments().at(Index).Type();

      // Disambiguate scalar and aggregate arguments.
      // Scalars are passed by value, aggregates - by pointer.
      Current.Type = ArgumentType;
      if (ArgumentType.isScalar())
        Current.Kind = ArgumentKind::Scalar;
      else
        Current.Kind = ArgumentKind::ReferenceToAggregate;

      Current.Registers = std::move(Args[Index].Registers);
      if (Args[Index].SizeOnStack != 0) {
        // TODO: further alignment considerations are needed here.
        Current.Stack = typename Layout::Argument::StackSpan{
          CurrentOffset, Args[Index].SizeOnStack
        };
        CurrentOffset += Args[Index].SizeOnStack;
      }
    }
    Result.CalleeSavedRegisters.resize(AT::CalleeSavedRegisters.size());
    llvm::copy(AT::CalleeSavedRegisters, Result.CalleeSavedRegisters.begin());

    Result.FinalStackOffset = ConversionHelper<A>::finalStackOffset(Args);

    return Result;
  })) {
}

Layout::Layout(const model::RawFunctionType &Function) {
  // Lay register arguments out.
  for (const model::NamedTypedRegister &Register : Function.Arguments()) {
    revng_assert(Register.Type().isScalar());

    auto &Argument = Arguments.emplace_back();
    Argument.Registers = { Register.Location() };
    Argument.Type = Register.Type();
    Argument.Kind = ArgumentKind::Scalar;
  }

  // Lay the return value out.
  for (const model::TypedRegister &Register : Function.ReturnValues()) {
    auto &ReturnValue = ReturnValues.emplace_back();
    ReturnValue.Registers = { Register.Location() };
    ReturnValue.Type = Register.Type();
  }

  // Lay stack arguments out.
  if (Function.StackArgumentsType().UnqualifiedType().isValid()) {
    const model::QualifiedType &StackArgType = Function.StackArgumentsType();
    // The stack argument, if present, should always be a struct.
    revng_assert(StackArgType.Qualifiers().empty());
    revng_assert(StackArgType.is(model::TypeKind::StructType));

    auto &Argument = Arguments.emplace_back();

    // Stack argument is always passed by pointer for RawFunctionType
    Argument.Type = StackArgType;
    Argument.Kind = ArgumentKind::ReferenceToAggregate;

    // Record the size
    const model::Type *OriginalStackType = StackArgType.UnqualifiedType().get();
    auto *StackStruct = llvm::cast<model::StructType>(OriginalStackType);
    if (StackStruct->Size() != 0)
      Argument.Stack = { 0, StackStruct->Size() };
  }

  // Fill callee saved registers.
  append(Function.PreservedRegisters(), CalleeSavedRegisters);

  // Set the final offset.
  FinalStackOffset = Function.FinalStackOffset();
}

bool Layout::verify() const {
  model::Architecture::Values ExpectedA = model::Architecture::Invalid;
  std::unordered_set<model::Register::Values> LookupHelper;
  auto VerificationHelper = [&](model::Register::Values Register) -> bool {
    // Ensure each register is present only once
    if (!LookupHelper.emplace(Register).second)
      return false;

    // Ensure all the registers belong to the same architecture
    if (ExpectedA == model::Architecture::Invalid)
      ExpectedA = model::Register::getReferenceArchitecture(Register);
    else if (ExpectedA != model::Register::getReferenceArchitecture(Register))
      return false;

    return true;
  };

  // Verify arguments
  LookupHelper.clear();
  for (model::Register::Values Register : argumentRegisters())
    if (!VerificationHelper(Register))
      return false;

  // Verify return values
  LookupHelper.clear();
  for (model::Register::Values Register : returnValueRegisters())
    if (!VerificationHelper(Register))
      return false;

  using namespace abi::FunctionType::ArgumentKind;
  auto SPTAR = ShadowPointerToAggregateReturnValue;
  bool SPTARFound = false;
  unsigned Index = 0;
  for (const auto &Argument : Arguments) {
    if (Argument.Kind == SPTAR) {
      // SPTAR must be the first argument
      if (Index != 0)
        return false;

      // There can be only one SPTAR
      if (SPTARFound)
        return false;

      // SPTAR needs to be associated to a single register
      if (Argument.Stack or Argument.Registers.size() != 1)
        return false;
    }

    ++Index;
  }

  return true;
}

size_t Layout::argumentRegisterCount() const {
  size_t Result = 0;

  for (const auto &Argument : Arguments)
    Result += Argument.Registers.size();

  return Result;
}

size_t Layout::returnValueRegisterCount() const {
  size_t Result = 0;

  for (const ReturnValue &ReturnValue : ReturnValues)
    Result += ReturnValue.Registers.size();

  return Result;
}

llvm::SmallVector<model::Register::Values, 8>
Layout::argumentRegisters() const {
  llvm::SmallVector<model::Register::Values, 8> Result;

  for (const auto &Argument : Arguments)
    Result.append(Argument.Registers.begin(), Argument.Registers.end());

  return Result;
}

llvm::SmallVector<model::Register::Values, 8>
Layout::returnValueRegisters() const {
  llvm::SmallVector<model::Register::Values, 8> Result;

  for (const ReturnValue &ReturnValue : ReturnValues)
    Result.append(ReturnValue.Registers.begin(), ReturnValue.Registers.end());

  return Result;
}

} // namespace abi::FunctionType
