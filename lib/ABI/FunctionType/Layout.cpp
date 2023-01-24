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
#include "revng/Model/Helpers.h"
#include "revng/Support/Debug.h"

static Logger Log("function-type-conversion-to-raw");

namespace abi::FunctionType {

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
  model::TypePath convert(const model::CABIFunctionType &FunctionType,
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

/// Helper for choosing a "generic" register type, mainly used for filling
/// typeless argument registers in.
///
/// @param Register The register for which the type is created.
/// @param Binary The binary containing the type.
static model::QualifiedType
genericRegisterType(model::Register::Values Register, model::Binary &Binary) {
  return model::QualifiedType{
    Binary.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                            model::Register::getSize(Register)),
    {}
  };
}

/// Helper for choosing the "Raw" type of a specific argument or return
/// value based on its "C" type and the register it is to be placed in.
///
/// \param ArgumentType The original "C" type.
/// \param Register Register this value is placed in.
/// \param Binary The binary object containing the type information.
///
/// \return The resulting "Raw" type for the current entity.
static model::QualifiedType chooseType(const model::QualifiedType &ArgumentType,
                                       const model::Register::Values Register,
                                       model::Binary &Binary) {
  std::optional<std::uint64_t> MaybeSize = ArgumentType.size();
  std::uint64_t TargetSize = model::Register::getSize(Register);

  if (!MaybeSize.has_value()) {
    return model::QualifiedType{
      Binary.getPrimitiveType(model::Register::primitiveKind(Register),
                              model::Register::getSize(Register)),
      {}
    };
  } else if (*MaybeSize > TargetSize) {
    // TODO: additional considerations should be made here
    //       for better position-based ABI support.
    return ArgumentType.getPointerTo(Binary.Architecture());
  } else if (!ArgumentType.isScalar()) {
    // TODO: this can be considerably improved, we can preserve more
    //       information here over just returning the default type.
    return model::QualifiedType{
      Binary.getPrimitiveType(model::Register::primitiveKind(Register),
                              model::Register::getSize(Register)),
      {}
    };
  }

  return ArgumentType;
}

model::TypePath
ToRawConverter::convert(const model::CABIFunctionType &FunctionType,
                        TupleTree<model::Binary> &Binary) const {
  revng_log(Log, "Converting a `CABIFunctionType` to `RawFunctionType`.");
  revng_log(Log, "Original type:\n" << serializeToString(FunctionType));
  LoggerIndent Indentation(Log);
  auto PointerQualifier = model::Qualifier::createPointer(ABI.getPointerSize());

  // Since this conversion cannot fail, nothing prevents us from creating
  // the result type right away.
  auto [NewType, NewTypePath] = Binary->makeType<model::RawFunctionType>();
  model::copyMetadata(NewType, FunctionType);

  // Since shadow arguments are a concern, we need to deal with the return
  // value first.
  auto ReturnValue = distributeReturnValue(FunctionType.ReturnType());
  if (!ReturnValue.Registers.empty()) {
    revng_assert(ReturnValue.SizeOnStack == 0);

    // The return value uses registers: pass them through to the new type.
    for (model::Register::Values Register : ReturnValue.Registers) {
      model::TypedRegister Converted;
      Converted.Location() = Register;

      const model::QualifiedType &ReturnType = FunctionType.ReturnType();
      Converted.Type() = ReturnValue.Registers.size() > 1 ?
                           genericRegisterType(Register, *Binary) :
                           chooseType(ReturnType, Register, *Binary);

      revng_log(Log,
                "Adding a return value register:\n"
                  << serializeToString(Register));

      NewType.ReturnValues().emplace(Converted);
    }
  } else if (ReturnValue.Size != 0) {
    // The return value uses a pointer-to-a-location: add it as an argument.

    auto MaybeReturnValueSize = FunctionType.ReturnType().size();
    revng_assert(MaybeReturnValueSize != std::nullopt);
    revng_assert(ReturnValue.Size == *MaybeReturnValueSize);

    model::QualifiedType ReturnType = FunctionType.ReturnType();
    ReturnType.Qualifiers().emplace_back(PointerQualifier);

    revng_assert(!ABI.GeneralPurposeReturnValueRegisters().empty());
    auto FirstRegister = ABI.GeneralPurposeReturnValueRegisters()[0];
    model::TypedRegister ReturnPointer(FirstRegister);
    ReturnPointer.Type() = std::move(ReturnType);
    NewType.ReturnValues().insert(std::move(ReturnPointer));

    revng_log(Log,
              "Return value is returned through a shadow-argument-pointer:\n"
                << serializeToString(ReturnType));
  } else {
    // The function returns `void`: no need to do anything special.
    revng_log(Log, "Return value is `void`\n");
  }

  // Now that return value is figured out, the arguments are next.
  auto Arguments = distributeArguments(FunctionType.Arguments(),
                                       ReturnValue.SizeOnStack != 0);

  model::StructType StackArguments;
  uint64_t CombinedStackArgumentSize = 0;
  for (size_t ArgIndex = 0; ArgIndex < Arguments.size(); ++ArgIndex) {
    auto &ArgumentStorage = Arguments[ArgIndex];
    const auto &ArgumentType = FunctionType.Arguments().at(ArgIndex).Type();

    // Transfer the register arguments first.
    if (!ArgumentStorage.Registers.empty()) {
      for (auto Register : ArgumentStorage.Registers) {
        model::NamedTypedRegister Argument(Register);

        if (ArgumentStorage.Registers.size() > 1) {
          Argument.Type() = genericRegisterType(Register, *Binary);
          revng_log(Log,
                    "Adding a register as a part of a bigger argument:\n"
                      << serializeToString(Register));
        } else {
          Argument.Type() = chooseType(ArgumentType, Register, *Binary);
          revng_log(Log,
                    "Adding a register argument:\n"
                      << serializeToString(Register));
        }

        NewType.Arguments().emplace(Argument);
      }
    }

    // Then stack arguments.
    if (ArgumentStorage.SizeOnStack != 0) {
      auto ArgumentIterator = FunctionType.Arguments().find(ArgIndex);
      revng_assert(ArgumentIterator != FunctionType.Arguments().end());
      const model::Argument &Argument = *ArgumentIterator;

      // Round the next offset based on the natural alignment.
      uint64_t Alignment = *ABI.alignment(Argument.Type());
      CombinedStackArgumentSize += (Alignment
                                    - CombinedStackArgumentSize % Alignment);

      // Each argument gets converted into a struct field.
      model::StructField Field;
      model::copyMetadata(Field, Argument);
      Field.Offset() = CombinedStackArgumentSize;
      Field.Type() = Argument.Type();
      StackArguments.Fields().emplace(std::move(Field));

      revng_log(Log, "Adding a stack argument:\n" << serializeToString(Field));

      // Compute the full size of the argument (including padding if needed),
      // so that the next argument is not placed into this occupied space.
      auto MaybeSize = Argument.Type().size();
      revng_assert(MaybeSize.has_value() && MaybeSize.value() != 0);
      CombinedStackArgumentSize += ABI.paddedSizeOnStack(MaybeSize.value());
    }
  }

  // If the stack argument struct is not empty, record it into the model.
  if (CombinedStackArgumentSize != 0) {
    StackArguments.Size() = CombinedStackArgumentSize;

    using namespace model;
    auto Type = UpcastableType::make<StructType>(std::move(StackArguments));
    NewType.StackArgumentsType() = { Binary->recordNewType(std::move(Type)),
                                     {} };
  }

  // Set the final stack offset
  NewType.FinalStackOffset() = finalStackOffset(Arguments);

  // Populate the list of preserved registers
  for (auto Inserter = NewType.PreservedRegisters().batch_insert();
       model::Register::Values Register : ABI.CalleeSavedRegisters())
    Inserter.insert(Register);

  revng_log(Log, "Conversion successful:\n" << serializeToString(NewType));

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceAllUsesWith(FunctionType.key(), NewTypePath, Binary);

  // And don't forget to remove the old type.
  Binary->Types().erase(FunctionType.key());

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
  Result.resize(Arguments.size());

  for (const model::Argument &Argument : Arguments) {
    std::size_t RegisterIndex = Argument.Index() + SkippedCount;
    revng_assert(Argument.Index() < Result.size());
    auto &Distributed = Result[Argument.Index()];

    auto MaybeSize = Argument.Type().size();
    revng_assert(MaybeSize.has_value());
    Distributed.Size = *MaybeSize;

    if (Argument.Type().isFloat()) {
      if (RegisterIndex < ABI.VectorArgumentRegisters().size()) {
        auto Register = ABI.VectorArgumentRegisters()[RegisterIndex];
        Distributed.Registers.emplace_back(Register);
      } else {
        Distributed.SizeOnStack = Distributed.Size;
      }
    } else {
      if (RegisterIndex < ABI.GeneralPurposeArgumentRegisters().size()) {
        auto Reg = ABI.GeneralPurposeArgumentRegisters()[RegisterIndex];
        Distributed.Registers.emplace_back(Reg);
      } else {
        Distributed.SizeOnStack = Distributed.Size;
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
  size_t UsedGeneralPurposeRegisterCounter = SkippedCount;
  size_t UsedVectorRegisterCounter = 0;

  for (const model::Argument &Argument : Arguments) {
    auto Size = Argument.Type().size();
    revng_assert(Size.has_value());

    const bool CanSplit = ABI.ArgumentsCanBeSplitBetweenRegistersAndStack();
    if (Argument.Type().isFloat()) {
      // The conventional non-position based approach is not applicable for
      // vector registers since it's rare for multiple registers to be used
      // to pass a single argument.
      //
      // For now, provide at most a single vector register for such a value,
      // if there's a free one.
      //
      // TODO: find reproducers and handle the cases where multiple vector
      //       registers are used together.
      std::size_t Limit = 1;

      size_t &Counter = UsedVectorRegisterCounter;
      const auto &Registers = ABI.VectorArgumentRegisters();
      auto [Distributed, NextIndex] = considerRegisters(*Size,
                                                        Limit,
                                                        Counter,
                                                        Registers,
                                                        false);
      revng_assert(NextIndex == Counter || NextIndex == Counter + 1);
      Counter = NextIndex;

      if (Result.size() <= Argument.Index())
        Result.resize(Argument.Index() + 1);
      Result[Argument.Index()] = Distributed;
    } else {
      const auto &Registers = ABI.GeneralPurposeArgumentRegisters();
      size_t &Counter = UsedGeneralPurposeRegisterCounter;
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
  bool SkippedRegisterCount = 0;

  if (HasReturnValueLocationArgument == true)
    if (const auto &GPRs = ABI.GeneralPurposeArgumentRegisters(); !GPRs.empty())
      if (ABI.ReturnValueLocationRegister() == GPRs[0])
        SkippedRegisterCount = 1;

  if (ABI.ArgumentsArePositionBased())
    return distributePositionBasedArguments(Arguments, SkippedRegisterCount);
  else
    return distributeNonPositionBasedArguments(Arguments, SkippedRegisterCount);
}

static constexpr auto UnlimitedRegisters = std::numeric_limits<size_t>::max();

using model::QualifiedType;
DistributedArgument
ToRawConverter::distributeReturnValue(const QualifiedType &ReturnValue) const {
  if (ReturnValue.isVoid())
    return DistributedArgument{};

  auto MaybeSize = ReturnValue.size();
  revng_assert(MaybeSize.has_value());

  if (ReturnValue.isFloat()) {
    const auto &Registers = ABI.VectorReturnValueRegisters();

    // For now replace unsupported floating point return values with `void`
    // The main offenders are the values returned in `st0`.
    // TODO: handle this properly.
    if (Registers.empty())
      return DistributedArgument{};

    // TODO: replace this the explicit single register limit with an abi-defined
    // value. For more information see the relevant comment in
    // `distributeRegisterArguments`.
    std::size_t Limit = 1;
    return considerRegisters(*MaybeSize, Limit, 0, Registers, false).first;
  } else {
    const size_t Limit = ReturnValue.isScalar() ?
                           ABI.MaximumGPRsPerScalarReturnValue() :
                           ABI.MaximumGPRsPerAggregateReturnValue();
    const auto &Registers = ABI.GeneralPurposeReturnValueRegisters();
    return considerRegisters(*MaybeSize, Limit, 0, Registers, false).first;
  }
}

model::TypePath convertToRaw(const model::CABIFunctionType &FunctionType,
                             TupleTree<model::Binary> &Binary) {
  ToRawConverter ToRaw(abi::Definition::get(FunctionType.ABI()));
  return ToRaw.convert(FunctionType, Binary);
}

Layout::Layout(const model::CABIFunctionType &Function) {
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  ToRawConverter Converter(ABI);

  std::size_t CurrentStackOffset = 0;
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

    // Add an argument to represent the pointer to the return value location.
    Argument &RVLocationIn = Arguments.emplace_back();
    RVLocationIn.Type = Function.ReturnType().getPointerTo(Architecture);
    RVLocationIn.Kind = ArgumentKind::ShadowPointerToAggregateReturnValue;

    if (ABI.ReturnValueLocationRegister() != model::Register::Invalid) {
      // Return value is passed using the stack (with a pointer to the location
      // in the dedicated register).
      RVLocationIn.Registers.emplace_back(ABI.ReturnValueLocationRegister());
    } else if (ABI.ReturnValueLocationOnStack()) {
      // The location, where return value should be put in, is also communicated
      // using the stack.
      CurrentStackOffset += model::ABI::getPointerSize(ABI.ABI());
      RVLocationIn.Stack = { 0, CurrentStackOffset };
    } else {
      revng_abort("Big return values are not supported by the current ABI");
    }

    // Also return the same pointer using normal means.
    //
    // NOTE: maybe some architectures do not require this.
    // TODO: investigate.
    ReturnValue &RVLocationOut = ReturnValues.emplace_back();
    RVLocationOut.Type = RVLocationIn.Type;
    revng_assert(RVLocationOut.Type.UnqualifiedType().isValid());

    // To simplify selecting the register for it, use the full distribution
    // routine again, but with the pointer instead of the original type.
    auto RVOut = Converter.distributeReturnValue(RVLocationOut.Type);
    revng_assert(RVOut.Size == model::ABI::getPointerSize(ABI.ABI()));
    revng_assert(RVOut.Registers.size() == 1);
    revng_assert(RVOut.SizeOnStack == 0);
    RVLocationOut.Registers = std::move(RVOut.Registers);
  }

  auto Args = Converter.distributeArguments(Function.Arguments(),
                                            RV.SizeOnStack != 0);
  revng_assert(Args.size() == Function.Arguments().size());
  for (size_t Index = 0; Index < Args.size(); ++Index) {
    Argument &Current = Arguments.emplace_back();
    const auto &ArgumentType = Function.Arguments().at(Index).Type();

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
        CurrentStackOffset, Args[Index].SizeOnStack
      };
      CurrentStackOffset += Args[Index].SizeOnStack;
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

  // Verify callee saved registers
  LookupHelper.clear();
  for (model::Register::Values Register : CalleeSavedRegisters)
    if (!VerificationHelper(Register))
      return false;

  using namespace abi::FunctionType::ArgumentKind;
  auto SPTAR = ShadowPointerToAggregateReturnValue;
  bool SPTARFound = false;
  bool IsFirst = true;
  for (const auto &Argument : Arguments) {
    if (Argument.Kind == SPTAR) {
      // SPTAR must be the first argument
      if (!IsFirst)
        return false;

      // There can be only one SPTAR
      if (SPTARFound)
        return false;

      if (Argument.Stack.has_value()) {
        // SPTAR can be on the stack if ABI allows that.
        //
        // TODO: we should probably verify that, but such a verification would
        //       require access to the ABI in question.

        revng_assert(ExpectedA != model::Architecture::Invalid,
                     "Unable to figure out the architecture.");
        auto PointerSize = model::Architecture::getPointerSize(ExpectedA);

        // The space SPTAR occupies on stack has to be that of a single pointer.
        // It also has to be the first argument (with offset equal to zero).
        if (Argument.Stack->Size != PointerSize || Argument.Stack->Offset != 0)
          return false;
      } else {
        // SPTAR is not on the stack, so it has to be a single register
        if (Argument.Registers.size() != 1)
          return false;
      }
    }

    IsFirst = false;
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
