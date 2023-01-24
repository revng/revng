/// \file Layout.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <span>
#include <unordered_set>

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Model/Binary.h"

namespace abi::FunctionType {

static model::QualifiedType
buildGenericType(model::Register::Values Register, model::Binary &Binary) {
  constexpr auto Kind = model::PrimitiveTypeKind::Generic;
  size_t Size = model::Register::getSize(Register);
  return model::QualifiedType(Binary.getPrimitiveType(Kind, Size), {});
}

using RegisterVector = llvm::SmallVector<model::Register::Values, 1>;
struct DistributedArgument {
  RegisterVector Registers = {};
  size_t Size = 0, SizeOnStack = 0;
};
using DistributedArguments = llvm::SmallVector<DistributedArgument, 4>;

using RegisterSpan = std::span<const model::Register::Values>;
using ArgumentSet = SortedVector<model::Argument>;

class ToRawConverter {
private:
  const abi::Definition &ABI;

public:
  explicit ToRawConverter(const abi::Definition &ABI) : ABI(ABI) {
    revng_assert(ABI.verify());
  }

public:
  /// Entry point for the `toRaw` conversion.
  model::TypePath convert(const model::CABIFunctionType &Function,
                          TupleTree<model::Binary> &Binary) const;

  /// Helper used for deciding how an arbitrary return type should be
  /// distributed across registers and the stack accordingly to the \ref ABI.
  ///
  /// \param ReturnValueType The model type that should be returned.
  /// \return Information about registers and stack that are to be used to
  ///         return the said type.
  DistributedArgument
  distributeReturnValue(const model::QualifiedType &ReturnValueType) const;

  /// Helper used for deciding how an arbitrary set of arguments should be
  /// distributed across registers and the stack accordingly to the \ref ABI.
  ///
  /// \param Arguments The list of arguments to distribute.
  /// \param PassesReturnValueLocationAsAnArgument `true` if the first argument
  ///        slot should be occupied by a shadow return value, `false` otherwise
  /// \return Information about registers and stack that are to be used to
  ///         pass said arguments.
  DistributedArguments
  distributeArguments(const ArgumentSet &Arguments,
                      bool PassesReturnValueLocationAsAnArgument) const;

private:
  DistributedArguments
  distributePositionBasedArguments(const ArgumentSet &Arguments,
                                   std::size_t SkippedCount = 0) const;
  DistributedArguments
  distributeNonPositionBasedArguments(const ArgumentSet &Arguments,
                                      std::size_t SkippedCount = 0) const;

  std::pair<DistributedArgument, size_t>
  considerRegisters(size_t Size,
                    size_t AllowedRegisterLimit,
                    size_t OccupiedRegisterCount,
                    RegisterSpan Registers,
                    bool AllowPuttingPartOfAnArgumentOnStack) const;

public:
  uint64_t finalStackOffset(const DistributedArguments &Arguments) const;
};

static model::QualifiedType
chooseArgumentType(const model::QualifiedType &ArgumentType,
                   model::Register::Values Register,
                   const RegisterVector &Registers,
                   model::Binary &Binary) {
  if (Registers.size() > 1) {
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

model::TypePath
ToRawConverter::convert(const model::CABIFunctionType &Function,
                        TupleTree<model::Binary> &Binary) const {
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
      CombinedStackArgumentSize += ABI.paddedSizeOnStack(MaybeSize.value());
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
        const model::Type *Type = Function.ReturnType().UnqualifiedType().get();
        revng_assert(Type != nullptr);
        const auto *Struct = llvm::dyn_cast<model::StructType>(Type);
        if (Struct && Struct->Fields().size() == Result.ReturnValues().size()) {
          using RegisterEnum = model::Register::Values;
          SmallMap<RegisterEnum, model::QualifiedType, 4> RecoveredTypes;
          size_t StructOffset = 0;
          for (size_t Index = 0; Index < Struct->Fields().size(); ++Index) {
            if (Index >= ABI.GeneralPurposeReturnValueRegisters().size())
              break;
            auto Register = ABI.GeneralPurposeReturnValueRegisters()[Index];

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
      revng_assert(!ABI.GeneralPurposeReturnValueRegisters().empty());
      auto Register = ABI.GeneralPurposeReturnValueRegisters()[0];
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
  for (model::Register::Values Register : ABI.CalleeSavedRegisters())
    Result.PreservedRegisters().insert(Register);

  // Steal the ID
  Result.ID() = Function.ID();

  // Add converted type to the model.
  using UT = model::UpcastableType;
  auto Ptr = UT::make<model::RawFunctionType>(std::move(Result));
  auto NewTypePath = Binary->recordNewType(std::move(Ptr));

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceAllUsesWith(Function.key(), NewTypePath, Binary);

  // And don't forget to remove the old type.
  Binary->Types().erase(Function.key());

  return NewTypePath;
}

uint64_t
ToRawConverter::finalStackOffset(const DistributedArguments &Arguments) const {
  const auto Architecture = model::ABI::getArchitecture(ABI.ABI());
  uint64_t Result = model::Architecture::getCallPushSize(Architecture);

  if (ABI.CalleeIsResponsibleForStackCleanup()) {
    for (auto &Argument : Arguments)
      Result += Argument.SizeOnStack;

    // TODO: take return values into the account.

    // TODO: take shadow space into the account if relevant.

    revng_assert(llvm::isPowerOf2_64(ABI.StackAlignment()));
    Result += ABI.StackAlignment() - 1;
    Result &= ~(ABI.StackAlignment() - 1);
  }

  return Result;
}

DistributedArguments
ToRawConverter::distributePositionBasedArguments(const ArgumentSet &Arguments,
                                                 size_t SkippedCount) const {
  DistributedArguments Result;

  for (const model::Argument &Argument : Arguments) {
    std::size_t RegisterIndex = Argument.Index() + SkippedCount;
    if (Result.size() <= RegisterIndex)
      Result.resize(RegisterIndex + 1);
    auto &Distributed = Result[RegisterIndex];

    auto MaybeSize = Argument.Type().size();
    revng_assert(MaybeSize.has_value());
    Distributed.Size = *MaybeSize;

    if (Argument.Type().isFloat()) {
      if (RegisterIndex < ABI.VectorArgumentRegisters().size()) {
        auto Register = ABI.VectorArgumentRegisters()[RegisterIndex];
        Distributed.Registers.emplace_back(Register);
      } else {
        Distributed.SizeOnStack = ABI.paddedSizeOnStack(Distributed.Size);
      }
    } else {
      if (RegisterIndex < ABI.GeneralPurposeArgumentRegisters().size()) {
        auto Reg = ABI.GeneralPurposeArgumentRegisters()[RegisterIndex];
        Distributed.Registers.emplace_back(Reg);
      } else {
        Distributed.SizeOnStack = ABI.paddedSizeOnStack(Distributed.Size);
      }
    }
  }

  return Result;
}

std::pair<DistributedArgument, std::size_t>
ToRawConverter::considerRegisters(std::size_t Size,
                                  std::size_t AllowedRegisterLimit,
                                  std::size_t OccupiedRegisterCount,
                                  RegisterSpan Registers,
                                  bool AllowPartOfAnArgumentOnTheStack) const {
  size_t RegisterLimit = OccupiedRegisterCount + AllowedRegisterLimit;
  size_t ConsideredRegisterCounter = OccupiedRegisterCount;

  size_t SizeCounter = 0;
  const size_t ARC = Registers.size();
  if (ARC > 0) {
    size_t &CRC = ConsideredRegisterCounter;
    while (SizeCounter < Size && CRC < ARC && CRC < RegisterLimit) {
      size_t RegisterIndex = ConsideredRegisterCounter++;
      auto CurrentRegister = Registers[RegisterIndex];
      SizeCounter += model::Register::getSize(CurrentRegister);
    }
  }

  DistributedArgument DA;
  DA.Size = Size;

  if (ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
    if (ConsideredRegisterCounter - OccupiedRegisterCount == 2) {
      if ((OccupiedRegisterCount & 1) != 0) {
        ++OccupiedRegisterCount;
        ++ConsideredRegisterCounter;
      }
    }
  }

  if (SizeCounter >= Size) {
    for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = 0;
  } else if (AllowPartOfAnArgumentOnTheStack) {
    for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = DA.Size - SizeCounter;
  } else {
    DA.SizeOnStack = DA.Size;
    ConsideredRegisterCounter = OccupiedRegisterCount;
  }

  if (DA.SizeOnStack != 0)
    DA.SizeOnStack = ABI.paddedSizeOnStack(DA.SizeOnStack);

  return { DA, ConsideredRegisterCounter };
}

using ASet = ArgumentSet;
DistributedArguments
ToRawConverter::distributeNonPositionBasedArguments(const ASet &Arguments,
                                                    size_t SkippedCount) const {
  DistributedArguments Result;
  size_t UsedGeneralPurposeRegisterCount = SkippedCount;
  size_t UsedVectorRegisterCount = 0;

  for (const model::Argument &Argument : Arguments) {
    auto Size = Argument.Type().size();
    revng_assert(Size.has_value());

    const bool CanSplit = ABI.ArgumentsCanBeSplitBetweenRegistersAndStack();
    if (Argument.Type().isFloat()) {
      // The conventional non-position based approach is not applicable for
      // vector registers since it's rare for multiple registers to be used
      // to pass a single argument.
      //
      // For now, just return the next free register.
      //
      // TODO: handle this properly.

      const auto &Registers = ABI.VectorArgumentRegisters();
      if (UsedVectorRegisterCount < Registers.size()) {
        // There is a free register to put the argument in.
        if (Result.size() <= Argument.Index())
          Result.resize(Argument.Index() + 1);

        auto Register = Registers[UsedVectorRegisterCount];
        Result[Argument.Index()].Registers.emplace_back(Register);
        Result[Argument.Index()].Size = *Size;
        Result[Argument.Index()].SizeOnStack = 0;

        UsedVectorRegisterCount++;
      } else {
        // There are no more free registers left,
        // pass the argument on the stack.
        if (Result.size() <= Argument.Index())
          Result.resize(Argument.Index() + 1);
        Result[Argument.Index()].Size = *Size;
        Result[Argument.Index()].SizeOnStack = ABI.paddedSizeOnStack(*Size);
      }
    } else {
      const auto &Registers = ABI.GeneralPurposeArgumentRegisters();
      size_t &Counter = UsedGeneralPurposeRegisterCount;
      if (Argument.Type().isScalar()) {
        const size_t Limit = ABI.MaximumGPRsPerScalarArgument();

        auto [Distributed, NextIndex] = considerRegisters(*Size,
                                                          Limit,
                                                          Counter,
                                                          Registers,
                                                          CanSplit);
        if (Result.size() <= Argument.Index())
          Result.resize(Argument.Index() + 1);
        Result[Argument.Index()] = Distributed;
        Counter = NextIndex;
      } else {
        const size_t Limit = ABI.MaximumGPRsPerAggregateArgument();

        auto [Distributed, NextIndex] = considerRegisters(*Size,
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

DistributedArguments
ToRawConverter::distributeArguments(const ArgumentSet &Arguments,
                                    bool HasReturnValueLocationArgument) const {
  bool SkippedRegisters = 0;
  if (HasReturnValueLocationArgument == true)
    if (const auto &GPRs = ABI.GeneralPurposeArgumentRegisters(); !GPRs.empty())
      if (ABI.ReturnValueLocationRegister() == GPRs[0])
        SkippedRegisters = 1;

  if (ABI.ArgumentsArePositionBased())
    return distributePositionBasedArguments(Arguments, SkippedRegisters);
  else
    return distributeNonPositionBasedArguments(Arguments, SkippedRegisters);
}

using model::QualifiedType;
DistributedArgument
ToRawConverter::distributeReturnValue(const QualifiedType &ReturnValue) const {
  if (ReturnValue.isVoid())
    return DistributedArgument{};

  auto MaybeSize = ReturnValue.size();
  revng_assert(MaybeSize.has_value());

  if (ReturnValue.isFloat()) {
    const auto &Registers = ABI.VectorReturnValueRegisters();

    // As a temporary measure always just return the first vector return value
    // register, if it's available.
    //
    // TODO: handle the situation properly.
    if (!ABI.VectorReturnValueRegisters().empty()) {
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
    const size_t L = ReturnValue.isScalar() ?
                       ABI.MaximumGPRsPerScalarReturnValue() :
                       ABI.MaximumGPRsPerAggregateReturnValue();
    const auto &Registers = ABI.GeneralPurposeReturnValueRegisters();
    return considerRegisters(*MaybeSize, L, 0, Registers, false).first;
  }
}

model::TypePath convertToRaw(const model::CABIFunctionType &Function,
                             TupleTree<model::Binary> &Binary) {
  ToRawConverter ToRaw(abi::Definition::get(Function.ABI()));
  return ToRaw.convert(Function, Binary);
}

Layout::Layout(const model::CABIFunctionType &Function) {
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  ToRawConverter Converter(ABI);

  const auto Architecture = model::ABI::getArchitecture(Function.ABI());
  auto RV = Converter.distributeReturnValue(Function.ReturnType());
  if (RV.SizeOnStack == 0) {
    // Nothing on the stack, the return value fits into the registers.
    auto &ReturnValue = ReturnValues.emplace_back();
    ReturnValue.Type = Function.ReturnType();
    ReturnValue.Registers = std::move(RV.Registers);
  } else {
    revng_assert(RV.Registers.empty(),
                 "Register and stack return values should never be present "
                 "at the same time.");
    revng_assert(ABI.ReturnValueLocationRegister() != model::Register::Invalid,
                 "Big return values are not supported by the current ABI");
    auto &RVLocationArg = Arguments.emplace_back();
    RVLocationArg.Registers.emplace_back(ABI.ReturnValueLocationRegister());
    RVLocationArg.Type = Function.ReturnType().getPointerTo(Architecture);
    RVLocationArg.Kind = ArgumentKind::ShadowPointerToAggregateReturnValue;
  }

  size_t CurrentOffset = 0;
  auto Args = Converter.distributeArguments(Function.Arguments(),
                                            RV.SizeOnStack != 0);
  revng_assert(Args.size() == Function.Arguments().size());
  for (size_t Index = 0; Index < Args.size(); ++Index) {
    auto &Current = Arguments.emplace_back();
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
  CalleeSavedRegisters.resize(ABI.CalleeSavedRegisters().size());
  llvm::copy(ABI.CalleeSavedRegisters(), CalleeSavedRegisters.begin());

  FinalStackOffset = Converter.finalStackOffset(Args);
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
