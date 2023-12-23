/// \file ValueDistributor.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

#include "ValueDistributor.h"

static Logger Log("function-type-value-distributor");

namespace abi::FunctionType {

std::pair<DistributedValues, uint64_t>
ValueDistributor::distribute(model::QualifiedType Type,
                             RegisterSpan Registers,
                             uint64_t OccupiedRegisterCount,
                             uint64_t AllowedRegisterLimit,
                             bool ForbidSplittingBetweenRegistersAndStack) {
  DistributedValues Result;

  LoggerIndent Indentation(Log);
  revng_log(Log, "Distributing a value between the registers and the stack.");
  revng_log(Log,
            OccupiedRegisterCount << " registers are already occupied, while "
                                     "only "
                                  << AllowedRegisterLimit
                                  << " registers are available to be used.");
  revng_log(Log, "The total number of registers is " << Registers.size());
  revng_log(Log, "The type:\n" << serializeToString(Type));

  uint64_t Size = *Type.size();
  abi::Definition::AlignmentCache Cache;
  uint64_t Alignment = *ABI.alignment(Type, Cache);
  bool HasNaturalAlignment = *ABI.hasNaturalAlignment(Type, Cache);

  revng_log(Log,
            "Its size is " << Size << " and its "
                           << (HasNaturalAlignment ? "" : "un")
                           << "natural alignment is " << Alignment << ".");

  bool CanUseRegisters = HasNaturalAlignment;
  if (ABI.AllowUnnaturallyAlignedTypesInRegisters())
    CanUseRegisters = true;

  // Precompute the last register allowed to be used.
  uint64_t LastRegister = OccupiedRegisterCount + AllowedRegisterLimit;
  if (LastRegister > Registers.size())
    LastRegister = Registers.size();

  // Define the counters: one for the number of registers the current value
  // could occupy, and one for the total size of said registers.
  uint64_t ConsideredRegisterCounter = OccupiedRegisterCount;
  uint64_t SizeCounter = 0;

  // Keep adding the registers until either the total size exceeds needed or
  // we run out of allowed registers.
  if (!Registers.empty()) {
    auto ShouldContinue = [&]() -> bool {
      return SizeCounter < Size && ConsideredRegisterCounter < Registers.size()
             && ConsideredRegisterCounter < LastRegister;
    };
    while (ShouldContinue()) {
      const auto CurrentRegister = Registers[ConsideredRegisterCounter++];
      SizeCounter += model::Register::getSize(CurrentRegister);
    }
  }

  if (SizeCounter >= Size) {
    if (ConsideredRegisterCounter - OccupiedRegisterCount == 1) {
      revng_log(Log, "A single register is sufficient to hold the value.");
    } else {
      revng_log(Log,
                (ConsideredRegisterCounter - OccupiedRegisterCount)
                  << " registers are sufficient to hold the value.");
    }
  }

  // Take the alignment into consideration on the architectures that require
  // padding to be inserted even for arguments passed in registers.
  if (ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
    const uint64_t PointerSize = ABI.getPointerSize();
    bool MultiAligned = (Size >= PointerSize && Alignment > PointerSize);
    bool LastRegisterUsed = ConsideredRegisterCounter == OccupiedRegisterCount;
    bool FirstRegisterOdd = (OccupiedRegisterCount & 1) != 0;
    if (MultiAligned && !LastRegisterUsed && FirstRegisterOdd) {
      LoggerIndent Indentation(Log);
      revng_log(Log,
                "Because the ABI requires arguments placed in the "
                "registers to also be aligned, an extra register needs "
                "to be used to hold the padding.");

      // Add an extra "padding" argument to represent this.
      DistributedValue &Padding = Result.emplace_back();
      Padding.Registers = { Registers[OccupiedRegisterCount++] };
      Padding.Size = model::Register::getSize(Padding.Registers[0]);
      Padding.RepresentsPadding = true;

      revng_assert(SizeCounter >= Padding.Size);
      SizeCounter -= Padding.Size;
      if (ConsideredRegisterCounter < LastRegister)
        ++ConsideredRegisterCounter;
    }
  }

  DistributedValue &DA = Result.emplace_back();
  DA.Size = Size;

  bool AllowSplitting = !ForbidSplittingBetweenRegistersAndStack
                        && ABI.ArgumentsCanBeSplitBetweenRegistersAndStack();
  if (SizeCounter >= Size && CanUseRegisters) {
    // This a register-only argument, add the registers.
    for (uint64_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = 0;
  } else if (AllowSplitting && ConsideredRegisterCounter == LastRegister) {
    // This argument is split among the registers and the stack.
    for (uint64_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = DA.Size - SizeCounter;
  } else {
    // This is a stack-only argument.
    DA.SizeOnStack = DA.Size;
    if (ABI.NoRegisterArgumentsCanComeAfterStackOnes()) {
      // Mark all the registers as occupied as soon as stack is used.
      ConsideredRegisterCounter = Registers.size();
    } else {
      // Leave registers unaffected, since the argument will only use stack.
      ConsideredRegisterCounter = OccupiedRegisterCount;
    }
  }

  // Don't forget to apply the tailing padding to the stack part of the argument
  // if it's present.
  if (DA.SizeOnStack != 0) {
    DA.SizeOnStack = ABI.paddedSizeOnStack(DA.SizeOnStack);
  }

  revng_log(Log, "Value successfully distributed.");
  LoggerIndent FurtherIndentation(Log);
  revng_log(Log,
            "It requires " << DA.Registers.size() << " registers, and "
                           << DA.SizeOnStack << " bytes on stack.");

  return { std::move(Result), ConsideredRegisterCounter };
}

DistributedValues
ArgumentDistributor::nonPositionBased(const model::QualifiedType &Type) {
  uint64_t RegisterLimit = 0;
  bool ForbidSplitting = false;
  uint64_t *RegisterCounter = nullptr;
  std::span<const model::Register::Values> RegisterList;
  if (Type.isFloat()) {
    RegisterList = ABI.VectorArgumentRegisters();
    RegisterCounter = &UsedVectorRegisterCount;

    if (RegisterList.size() > *RegisterCounter) {
      // The conventional non-position based approach is not applicable for
      // vector registers since it's rare for multiple registers to be used
      // to pass a single argument.
      //
      // For now, provide at most a single vector register for such a value,
      // if there's a free one.
      //
      // TODO: find reproducers and handle the cases where multiple vector
      //       registers are used together.
      DistributedValue Result;
      Result.Registers.emplace_back(RegisterList[(*RegisterCounter)++]);
      return { Result };
    } else {
      // If there are no free registers left, explicitly set the limit to 0,
      // so that the default argument distribution routine puts it on
      // the stack.
      RegisterLimit = 0;
    }

    // Explicitly disallow splitting vector arguments across the registers
    // and the stack.
    ForbidSplitting = true;
  } else {
    RegisterList = ABI.GeneralPurposeArgumentRegisters();
    RegisterCounter = &UsedGeneralPurposeRegisterCount;
    RegisterLimit = Type.isScalar() ? ABI.MaximumGPRsPerScalarArgument() :
                                      ABI.MaximumGPRsPerAggregateArgument();
  }

  auto [Result, NextRegisterIndex] = distribute(Type,
                                                RegisterList,
                                                *RegisterCounter,
                                                RegisterLimit,
                                                ForbidSplitting);

  // Verify that the next register makes sense.
  auto VerifyNextRegisterIndex = [&](uint64_t Current, uint64_t Next) {
    if (Current == Next)
      return true; // No registers were used for this argument.

    if (Next >= Current && Next <= Current + RegisterLimit)
      return true; // It's within the expected boundaries.

    if (Next == RegisterList.size()) {
      // All the register are marked as used. Only allow this on ABIs that
      // don't allow register arguments to come after stack ones.
      return ABI.NoRegisterArgumentsCanComeAfterStackOnes();
    }

    return false;
  };
  revng_assert(VerifyNextRegisterIndex(*RegisterCounter, NextRegisterIndex));
  *RegisterCounter = NextRegisterIndex;

  ++ArgumentIndex;
  return std::move(Result);
}

DistributedValues
ArgumentDistributor::positionBased(const model::QualifiedType &ArgumentType) {
  std::array Helper{ UsedGeneralPurposeRegisterCount,
                     UsedVectorRegisterCount,
                     ArgumentIndex };
  uint64_t Index = *std::ranges::max_element(Helper);

  DistributedValue Result;
  Result.Size = *ArgumentType.size();

  const auto &UsedRegisterList = ArgumentType.isFloat() ?
                                   ABI.VectorArgumentRegisters() :
                                   ABI.GeneralPurposeArgumentRegisters();
  uint64_t &UsedRegisterCount = ArgumentType.isFloat() ?
                                  UsedVectorRegisterCount :
                                  UsedGeneralPurposeRegisterCount;

  if (Index < UsedRegisterList.size()) {
    Result.Registers.emplace_back(UsedRegisterList[Index]);

    revng_assert(UsedRegisterCount <= Index);
    UsedRegisterCount = Index + 1;
  } else {
    Result.SizeOnStack = Result.Size;
  }

  ++ArgumentIndex;
  return { Result };
}

DistributedValues
ArgumentDistributor::nextArgument(const model::QualifiedType &Type) {
  if (ABI.ArgumentsArePositionBased())
    return positionBased(Type);
  else
    return nonPositionBased(Type);
}

DistributedValue
ReturnValueDistributor::returnValue(const model::QualifiedType &Type) {
  if (Type.isVoid())
    return DistributedValue{};

  uint64_t Limit = 0;
  std::span<const model::Register::Values> RegisterList;
  if (Type.isFloat()) {
    RegisterList = ABI.VectorReturnValueRegisters();

    // For now replace unsupported floating point return values with `void`
    // The main offenders are the values returned in `st0`.
    // TODO: handle this properly.
    if (RegisterList.empty())
      return DistributedValue{};

    // TODO: replace this the explicit single register limit with an abi-defined
    // value. For more information see the relevant comment in
    // `distributeRegisterArguments`.
    Limit = 1;
  } else {
    RegisterList = ABI.GeneralPurposeReturnValueRegisters();
    Limit = Type.isScalar() ? ABI.MaximumGPRsPerScalarReturnValue() :
                              ABI.MaximumGPRsPerAggregateReturnValue();
  }

  auto [Result, _] = distribute(Type, RegisterList, 0, Limit, true);
  revng_assert(Result.size() == 1, "Return values should never be padded.");
  return Result[0];
}

} // namespace abi::FunctionType
