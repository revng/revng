//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Binary.h"
#include "revng/Support/AlignmentHelpers.h"
#include "revng/Support/Debug.h"

#include "revng/Model/ABI/Generated/Early/FloatingPointStrategy.h"
#include "revng/Model/Generated/Early/Register.h"

#include "ValueDistributor.h"

static Logger Log("function-type-value-distributor");

namespace abi::FunctionType {

// TODO: currently the portion information is not preserved between arguments.
//
// Specifically, each `distribute()` call is supplied with `fullPortions()`
// version of the available register list.
//
// This is sufficient for cases where arguments do not share registers.
//
// However, when we begin handling ARM-32, whose floating point arguments pack
// together, leading to a single register used for multiple arguments, we would
// need to extend the distributor state to keep track of which portion of which
// register is already in use.

namespace {

/// Builds a portion that covers each of the registers in its entirety.
RegisterPortions
fullPortions(std::span<const model::Register::Values> Registers) {
  RegisterPortions Result;
  Result.reserve(Registers.size());

  for (model::Register::Values Register : Registers)
    Result.emplace_back(Register, model::Register::getSize(Register));

  return Result;
}

template<model::AnyType AnyType>
RecursiveCoroutine<std::optional<HomogeneousFloatAggregate>>
computeHomogeneousFloatAggregate(const AnyType &Type,
                                 uint64_t MaximumFieldCount) {
  if constexpr (std::derived_from<AnyType, model::TypeDefinition>) {
    if (const auto *S = llvm::dyn_cast<model::StructDefinition>(&Type)) {
      std::optional<HomogeneousFloatAggregate> Result;

      uint64_t ExpectedOffset = 0;
      for (const model::StructField &Field : S->Fields()) {
        // Any hole in-between the fields means padding, which disqualifies
        // the aggregate.
        if (Field.Offset() != ExpectedOffset)
          rc_return std::nullopt;

        auto FieldInfo = rc_recur
          computeHomogeneousFloatAggregate(*Field.Type(), MaximumFieldCount);
        if (not FieldInfo)
          rc_return std::nullopt;

        // All the leaf field have to be of the same floating point type.
        if (Result and Result->FieldSize != FieldInfo->FieldSize)
          rc_return std::nullopt;

        if (not Result)
          Result = FieldInfo;
        else
          Result->FieldCount += FieldInfo->FieldCount;

        if (Result->FieldCount > MaximumFieldCount)
          rc_return std::nullopt;

        ExpectedOffset += *Field.Type()->size();
      }

      // Any trailing padding also disqualifies the aggregate.
      if (not Result or ExpectedOffset != S->Size())
        rc_return std::nullopt;

      rc_return Result;

    } else if (const auto *U = llvm::dyn_cast<model::UnionDefinition>(&Type)) {
      std::optional<HomogeneousFloatAggregate> Result;
      for (const model::UnionField &Field : U->Fields()) {
        auto FieldInfo = rc_recur
          computeHomogeneousFloatAggregate(*Field.Type(), MaximumFieldCount);
        if (not FieldInfo)
          rc_return std::nullopt;

        // All the leaf fields have to be of the same floating point type.
        if (Result and Result->FieldSize != FieldInfo->FieldSize)
          rc_return std::nullopt;

        // The widest union field dictates its final member count.
        if (not Result or Result->FieldCount < FieldInfo->FieldCount)
          Result = FieldInfo;
      }

      // Any trailing padding also disqualifies the aggregate.
      if (Result)
        if (*Type.size() != Result->FieldSize * Result->FieldCount)
          rc_return std::nullopt;

      rc_return Result;

    } else if (auto *TD = llvm::dyn_cast<model::TypedefDefinition>(&Type)) {
      rc_return rc_recur computeHomogeneousFloatAggregate(*TD->UnderlyingType(),
                                                          MaximumFieldCount);

    } else {
      // Enums and function types do not have fields.
      revng_assert(llvm::isa<model::EnumDefinition>(&Type)
                     or llvm::isa<model::RawFunctionDefinition>(&Type)
                     or llvm::isa<model::CABIFunctionDefinition>(&Type),
                   "Unsupported type definition");

      rc_return std::nullopt;
    }

  } else if constexpr (std::derived_from<AnyType, model::Type>) {
    if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
      rc_return rc_recur computeHomogeneousFloatAggregate(D->unwrap(),
                                                          MaximumFieldCount);

    } else if (const auto *A = Type.getArray()) {
      auto Elem = rc_recur computeHomogeneousFloatAggregate(*A->ElementType(),
                                                            MaximumFieldCount);
      if (not Elem)
        rc_return std::nullopt;

      const uint64_t FieldCount = Elem->FieldCount * A->ElementCount();
      if (FieldCount > MaximumFieldCount)
        rc_return std::nullopt;

      rc_return HomogeneousFloatAggregate{ .FieldSize = Elem->FieldSize,
                                           .FieldCount = FieldCount };

    } else if (const auto *P = Type.getPrimitive()) {
      if (P->PrimitiveKind() != model::PrimitiveKind::Float)
        rc_return std::nullopt;

      rc_return HomogeneousFloatAggregate{ .FieldSize = *Type.size(),
                                           .FieldCount = 1 };

    } else if (Type.getPointer()) {
      // No pointers are allowed in HFAs.
      rc_return std::nullopt;
    }

    revng_abort("Unsupported type");

  } else {
    static_assert(type_always_false_v<AnyType>, "Unsupported type.");
  }
}

} // namespace

std::optional<HomogeneousFloatAggregate>
asHomogeneousFloatAggregate(const model::Type &Type,
                            uint64_t MaximumFieldCount) {
  return rc_eval(computeHomogeneousFloatAggregate(Type, MaximumFieldCount));
}

std::optional<HomogeneousFloatAggregate>
asHomogeneousFloatAggregate(const model::TypeDefinition &Type,
                            uint64_t MaximumFieldCount) {
  return rc_eval(computeHomogeneousFloatAggregate(Type, MaximumFieldCount));
}

std::pair<DistributedValues, uint64_t>
ValueDistributor::distribute(uint64_t Size,
                             uint64_t Alignment,
                             bool HasNaturalAlignment,
                             RegisterPortionSpan Registers,
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
  revng_log(Log,
            "Its size is " << Size << " and its "
                           << (HasNaturalAlignment ? "" : "un")
                           << "natural alignment is " << Alignment << ".");

  bool CanUseRegisters = HasNaturalAlignment;
  if (ABI.AllowPackedTypesInRegisters())
    CanUseRegisters = true;

  // Precompute the last register allowed to be used.
  uint64_t LastRegister = OccupiedRegisterCount + AllowedRegisterLimit;
  if (LastRegister > Registers.size())
    LastRegister = Registers.size();

  // Define the counters: one for the number of registers the current value
  // could occupy, and one for the total size of said registers.
  uint64_t ConsideredRegisterCounter = OccupiedRegisterCount;
  uint64_t SizeCounter = 0;

  // Take the alignment into consideration on the architectures that require
  // padding to be inserted even for arguments passed in registers.
  //
  // For example:
  // ```cpp
  // // compiled for arm-32
  // void test(uint32_t r0, uint32_t r1, uint32_t r2,
  //           uint64_t tested_argument);
  // ```
  // what would happen is that after the `r#` arguments are placed in
  // the corresponding registers, because the `tested_argument` is aligned
  // at 8 bytes, it must start from an even-numbered register (which `r3` is
  // not) so the *full* argument should be placed on the stack, with `r3`
  // becoming a padding register (it must be considered occupied).
  //
  // Specifically, if there are more arguments after that, like
  // ```cpp
  // // compiled for arm-32
  // void test(uint32_t r0, uint32_t r1, uint32_t r2,
  //           uint64_t tested_argument,
  //           uint32_t next_argument);
  // ```
  //
  // Even though `r3` is enough for `next_argument` *and* this is
  // a non-position-based ABI, it must not be allocated there, as `r3` is
  // already occupied with the padding.
  //
  // To make this work, we have to handle
  // the `OnlyStartDoubleArgumentsFromAnEvenRegister`
  // logic *before* doing any real distribution.
  if (ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
    const uint64_t PointerSize = ABI.getPointerSize();
    bool MultiAligned = Size >= PointerSize && Alignment > PointerSize;
    bool FirstRegisterOdd = (OccupiedRegisterCount & 1) != 0;
    if (MultiAligned && FirstRegisterOdd) {
      LoggerIndent Indentation(Log);
      revng_log(Log,
                "Because the ABI requires arguments placed in the "
                "registers to also be aligned, an extra register needs "
                "to be used to hold the padding.");

      if (OccupiedRegisterCount + 1 >= Registers.size()) {
        // Omit marking the very last register as padding since the argument
        // is going to end up on the stack anyway.

      } else {
        revng_assert(OccupiedRegisterCount + 1 < Registers.size());

        // Add an extra "padding" argument to represent this.
        DistributedValue &Padding = Result.emplace_back();
        Padding.Registers = { Registers[OccupiedRegisterCount] };
        Padding.Size = PointerSize;
        Padding.RepresentsPadding = true;

        ++OccupiedRegisterCount;

        // The padding register does not count towards the limit
        ++LastRegister;
      }

      ConsideredRegisterCounter = OccupiedRegisterCount;
    }
  }

  // Keep adding the registers until either the total size exceeds needed or
  // we run out of allowed registers.
  if (!Registers.empty()) {
    auto ShouldContinue = [&]() -> bool {
      return SizeCounter < Size && ConsideredRegisterCounter < Registers.size()
             && ConsideredRegisterCounter < LastRegister;
    };
    while (ShouldContinue()) {
      SizeCounter += Registers[ConsideredRegisterCounter].Size;
      ++ConsideredRegisterCounter;
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

  DistributedValue &DA = Result.emplace_back();
  DA.Size = Size;

  bool AllowSplitting = CanUseRegisters
                        && !ForbidSplittingBetweenRegistersAndStack
                        && ABI.ArgumentsCanBeSplitBetweenRegistersAndStack();
  revng_assert(ConsideredRegisterCounter <= Registers.size());
  bool AllTheRegistersAreInUse = ConsideredRegisterCounter == LastRegister;
  if (SizeCounter >= Size && CanUseRegisters) {
    // This a register-only argument, reserve portions of size matching
    // the exact number of bytes the value takes up.
    //
    // NOTE: As of now, the distribution results in each register only used
    //       *once* for the entire argument list.
    //       This will change in the future, once we add support for ARM-32
    //       floating point arguments.
    uint64_t SizeLeft = Size;
    for (auto I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I) {
      const model::Register::Portion &Available = Registers[I];
      const uint64_t PortionSize = std::min(Available.Size, SizeLeft);
      revng_assert(PortionSize != 0);

      DA.Registers.emplace_back(Available.Register, PortionSize);
      SizeLeft -= PortionSize;
    }

    revng_assert(SizeLeft == 0);
    DA.SizeOnStack = 0;
  } else if (AllowSplitting && AllTheRegistersAreInUse && SizeCounter > 0) {
    // This argument is split among the registers and the stack.
    for (uint64_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = ABI.paddedSizeOnStack(DA.Size - SizeCounter);
    DA.PostPaddingSize = DA.SizeOnStack - DA.Size + SizeCounter;
    DA.OffsetOnStack = UsedStackOffset;
  } else {
    // This is a stack-only argument.
    if (not HasNaturalAlignment) {
      Alignment = ABI.MinimumStackArgumentSize();
    }

    if (ABI.PackStackArguments() && HasNaturalAlignment) {
      UsedStackOffset -= std::min(UsedStackOffset, LastAddedStackPadding);
    }

    uint64_t AdjustedAlignment = std::max(CurrentStackAlignment, Alignment);
    uint64_t PrePaddedOffset = llvm::alignToPowerOf2(UsedStackOffset,
                                                     AdjustedAlignment);
    revng_assert(PrePaddedOffset >= UsedStackOffset);
    DA.PrePaddingSize = PrePaddedOffset - UsedStackOffset;
    DA.OffsetOnStack = UsedStackOffset = PrePaddedOffset;

    DA.SizeOnStack = ABI.paddedSizeOnStack(DA.Size);
    DA.PostPaddingSize = DA.SizeOnStack - DA.Size;

    if (ABI.NoRegisterArgumentsCanComeAfterStackOnes()) {
      // Mark all the registers as occupied as soon as stack is used.
      ConsideredRegisterCounter = Registers.size();
    } else {
      // Leave registers unaffected, since the argument will only use stack.
      ConsideredRegisterCounter = OccupiedRegisterCount;
    }
  }

  UsedStackOffset += DA.SizeOnStack;
  CurrentStackAlignment = Alignment;
  LastAddedStackPadding = DA.PostPaddingSize;

  revng_log(Log, "Value successfully distributed.");
  LoggerIndent FurtherIndentation(Log);
  if (Log.isEnabled()) {
    std::string Message = "It requires " + std::to_string(DA.Registers.size())
                          + " registers";
    if (!DA.Registers.empty()) {
      Message += " (";
      constexpr llvm::StringRef Separator = ", ";
      for (const model::Register::Portion &Portion : DA.Registers)
        Message += model::Register::getRegisterName(Portion.Register).str()
                   + Separator.str();
      Message.resize(Message.size() - Separator.size());
      Message += ")";
    }
    Message += ", and " + std::to_string(DA.SizeOnStack) + " bytes at offset "
               + std::to_string(DA.OffsetOnStack) + " of the stack.\n";
    revng_log(Log, std::move(Message));

    Message = "Total size is " + std::to_string(DA.Size) + ", which includes "
              + std::to_string(DA.PrePaddingSize) + " bytes of pre-padding and "
              + std::to_string(DA.PostPaddingSize) + " bytes of post-padding.";
    revng_log(Log, std::move(Message));

    Message = std::string("Pointer-to-copy mechanism ")
              + (DA.UsesPointerToCopy ? "is" : "is not") + " used, and it "
              + (DA.RepresentsPadding ? "represents" : "does not represent")
              + " padding.";
    revng_log(Log, std::move(Message));
  }

  return { std::move(Result), ConsideredRegisterCounter };
}

using MaybeHFA = std::optional<HomogeneousFloatAggregate>;
DistributedValues
ArgumentDistributor::nonPositionBased(bool IsScalar,
                                      FloatingPointKind FPKind,
                                      uint64_t Size,
                                      uint64_t Alignment,
                                      bool HasNaturalAlignment,
                                      const MaybeHFA &HFA) {
  auto Strategy = argumentStrategy(ABI, FPKind);
  if (Strategy == abi::FloatingPointStrategy::Ignored) {
    revng_log(Log,
              "Ignoring an argument because the floating point strategy of `"
                << model::ABI::getName(ABI.ABI()) << "` is `Ignored`.");

    ++ArgumentIndex;
    return { DistributedValue::ignoredValuePlaceholder(Size) };
  }

  // Adjust the strategy for ABIs that only allow the dedicated floating point
  // register set to be used for the aggregates that consist solely of floating
  // point members of the same type.
  if (not IsScalar and ABI.MaximumHFAFieldCount() != 0
      and not HFA.has_value()) {
    namespace FPS = abi::FloatingPointStrategy;
    if (Strategy == FPS::NextRegisterFromTheDedicatedSet)
      Strategy = FPS::NextRegisterFromTheGeneralPurposeRegisterSet;
  }

  uint64_t RegisterLimit = 0;
  bool ForbidSplitting = false;
  bool UsesPointerToCopy = false;
  uint64_t *RegisterCounter = nullptr;
  std::span<const model::Register::Values> RegisterList;
  if (Strategy == abi::FloatingPointStrategy::NextRegisterFromTheDedicatedSet) {
    RegisterList = ABI.FloatArgumentRegisters();
    RegisterCounter = &UsedFloatRegisterCount;

    if (HFA.has_value()) {
      // Every field takes a register of its own, so the aggregate needs as
      // many registers as it has fields.
      RegisterLimit = HFA->FieldCount;

    } else if (RegisterList.size() > *RegisterCounter) {
      DistributedValue Result;
      Result.Size = Size;

      const auto Register = RegisterList[*RegisterCounter];
      const uint64_t RegisterSize = model::Register::getSize(Register);
      revng_assert(Size <= RegisterSize);
      Result.Registers.emplace_back(Register, Size);

      ++(*RegisterCounter);
      ++ArgumentIndex;
      return { Result };

    } else {
      // If there are no free registers left, explicitly set the limit to 0,
      // so that the default argument distribution routine puts it on
      // the stack.
      RegisterLimit = 0;
    }

    // Explicitly disallow splitting float arguments across the registers
    // and the stack.
    ForbidSplitting = true;

  } else {
    RegisterList = ABI.GeneralPurposeArgumentRegisters();
    RegisterCounter = &UsedGeneralPurposeRegisterCount;

    if (Strategy == abi::FloatingPointStrategy::Stack) {
      // The value is passed directly on the stack: it is never offered a
      // register, and it is not split between the registers and the stack
      // either.
      RegisterLimit = 0;
      ForbidSplitting = true;

    } else {
      RegisterLimit = IsScalar ? ABI.MaximumGPRsPerScalarArgument() :
                                 ABI.MaximumGPRsPerAggregateArgument();
    }

    // Allow positional-ABI-like pointer-to-copy substitution if ABI requires it
    if (ABI.BigArgumentsUsePointersToCopy()
        && Size > ABI.getPointerSize() * RegisterLimit) {
      Size = ABI.getPointerSize();
      Alignment = ABI.getPointerSize();
      HasNaturalAlignment = true;
      UsesPointerToCopy = true;
    }
  }

  if (ABI.TreatAllAggregatesAsPacked() && !IsScalar) {
    // This is a bit of trick: manually setting the alignment to unnatural
    // forces the underlying distribution to be *extra* careful about how it
    // handles the type. Which just happened to be exactly what we want in
    // some other, unrelated to unnatural alignment, cases.
    HasNaturalAlignment = false;
  }

  RegisterPortions Portions = fullPortions(RegisterList);
  if (HFA.has_value())
    for (model::Register::Portion &Portion : Portions)
      Portion.Size = HFA->FieldSize;

  auto &&[Result, NextRegisterIndex] = distribute(Size,
                                                  Alignment,
                                                  HasNaturalAlignment,
                                                  Portions,
                                                  *RegisterCounter,
                                                  RegisterLimit,
                                                  ForbidSplitting);
  if (UsesPointerToCopy) {
    revng_assert(Result.size() == 1);
    Result[0].UsesPointerToCopy = UsesPointerToCopy;
  }

  // Verify that the next register makes sense.
  auto VerifyNextRegisterIndex = [&](uint64_t Current, uint64_t Next) {
    if (Current == Next)
      return true; // No registers were used for this argument.

    uint64_t PaddingByteCount = 0;
    for (auto &Distributed : Result)
      if (Distributed.RepresentsPadding)
        PaddingByteCount += Distributed.Registers.size();

    if (Next >= Current && Next - PaddingByteCount <= Current + RegisterLimit)
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

DistributedValues ArgumentDistributor::positionBased(FloatingPointKind FPKind,
                                                     uint64_t Size) {
  // Propagate ignored arguments up.
  const auto Strategy = argumentStrategy(ABI, FPKind);
  if (Strategy == abi::FloatingPointStrategy::Ignored) {
    revng_log(Log,
              "Ignoring an argument because the floating point strategy of `"
                << model::ABI::getName(ABI.ABI()) << "` is `Ignored`.");

    ++ArgumentIndex;

    // The slot is consumed even if an argument is ignored: that's the big
    // advantage of position-based ABIs - predictability over efficiency.
    uint64_t CurrentArgumentIndex = nextPositionBasedIndex();
    UsedGeneralPurposeRegisterCount = CurrentArgumentIndex + 1;

    return { DistributedValue::ignoredValuePlaceholder(Size) };
  }

  namespace FPS = abi::FloatingPointStrategy;
  bool UseFloatRegisters = Strategy == FPS::NextRegisterFromTheDedicatedSet
                           and FPKind == FloatingPointKind::FloatPrimitive;

  DistributedValue Result;
  Result.Size = Size;

  // Handle pointer-to-copy substitution.
  Result.UsesPointerToCopy = false;
  if (UseFloatRegisters) {
    revng_assert(not ABI.FloatArgumentRegisters().empty(),
                 "Using the dedicated floating point register set requires at "
                 "least one such register present!");
    const auto FirstRegister = ABI.FloatArgumentRegisters().front();
    if (Result.Size > model::Register::getSize(FirstRegister))
      Result.UsesPointerToCopy = true;

  } else {
    if (not ABI.GeneralPurposeArgumentRegisters().empty()) {
      const auto FirstGPR = ABI.GeneralPurposeArgumentRegisters().front();
      revng_assert(model::Register::getSize(FirstGPR) == ABI.getPointerSize(),
                   "GPR of an unsupported size is used as an argument.");
    }

    if (Result.Size > ABI.getPointerSize())
      Result.UsesPointerToCopy = true;
  }

  // Under a position based ABI every argument occupies one slot, so a value
  // that never goes in a register cannot simply be appended to the stack:
  // instead a pointer to copy is used.
  if (Strategy == abi::FloatingPointStrategy::Stack)
    Result.UsesPointerToCopy = true;

  if (Result.UsesPointerToCopy) {
    Result.Size = ABI.getPointerSize();

    // Pointers never use float registers.
    UseFloatRegisters = false;
  }

  const auto &UsedRegisters = UseFloatRegisters ?
                                ABI.FloatArgumentRegisters() :
                                ABI.GeneralPurposeArgumentRegisters();
  uint64_t &UsedRegisterCount = UseFloatRegisters ?
                                  UsedFloatRegisterCount :
                                  UsedGeneralPurposeRegisterCount;

  uint64_t CurrentArgumentIndex = nextPositionBasedIndex();
  if (CurrentArgumentIndex < UsedRegisters.size()) {
    const auto Register = UsedRegisters[CurrentArgumentIndex];
    revng_assert(Result.Size <= model::Register::getSize(Register));
    Result.Registers.emplace_back(Register, Result.Size);

    revng_assert(UsedRegisterCount <= CurrentArgumentIndex);
    UsedRegisterCount = CurrentArgumentIndex + 1;
  } else {
    if (UsedStackOffset) {
      if (ABI.paddedSizeOnStack(UsedStackOffset) != UsedStackOffset) {
        std::string Error = "Position-based stack does not support unaligned "
                            "stack offsets: "
                            + std::to_string(UsedStackOffset);
        revng_abort(Error.c_str());
      }
    }

    Result.OffsetOnStack = UsedStackOffset;
    Result.PrePaddingSize = 0;
    Result.PostPaddingSize = 0;

    Result.SizeOnStack = ABI.getPointerSize();
    UsedStackOffset += Result.SizeOnStack;
  }

  ++ArgumentIndex;
  return { Result };
}

DistributedValue ReturnValueDistributor::returnValue(const model::Type &T) {
  revng_assert(!T.isVoidPrimitive());

  // Position-based ABIs do not use floating point registers for aggregates:
  // such values are returned in the general purpose registers instead.
  auto FPKind = floatingPointKind(T);
  auto Strategy = returnValueStrategy(ABI, FPKind);
  if (Strategy == abi::FloatingPointStrategy::NextRegisterFromTheDedicatedSet
      and FPKind == FloatingPointKind::FloatContainingAggregate
      and ABI.ArgumentsArePositionBased()) {
    namespace FPS = abi::FloatingPointStrategy;
    Strategy = FPS::NextRegisterFromTheGeneralPurposeRegisterSet;
  }

  // Adjust the strategy for ABIs that only allow the dedicated floating point
  // register set to be used for the aggregates that consist solely of floating
  // point members of the same type.
  std::optional<HomogeneousFloatAggregate> HFA;
  if (ABI.MaximumHFAFieldCount() != 0 and not T.isScalar()) {
    namespace FPS = abi::FloatingPointStrategy;
    if (Strategy == FPS::NextRegisterFromTheDedicatedSet) {
      HFA = asHomogeneousFloatAggregate(T, ABI.MaximumHFAFieldCount());
      if (not HFA.has_value()) {
        // If it's not a homogeneous aggregate, it should be treated as
        // a non-float-containing aggregate.
        Strategy = FPS::NextRegisterFromTheGeneralPurposeRegisterSet;
      }
    }
  }

  if (Strategy == abi::FloatingPointStrategy::Ignored) {
    revng_log(Log,
              "Dropping a return value because the floating point strategy of `"
                << model::ABI::getName(ABI.ABI()) << "` is `Ignored`.");
    return DistributedValue::ignoredValuePlaceholder(*T.size());
  }

  uint64_t Limit = 0;
  std::span<const model::Register::Values> RegisterList;
  namespace FPS = abi::FloatingPointStrategy;
  if (Strategy == FPS::Stack) {
    // `Stack` in the return value context just means SPTAR.
    RegisterList = {};
    Limit = 0;

  } else if (Strategy == FPS::NextRegisterFromTheDedicatedSet) {
    RegisterList = ABI.FloatReturnValueRegisters();

    // For now replace unsupported floating point return values with `void`
    // The main offenders are the values returned in `st0`.
    // TODO: handle this properly.
    if (RegisterList.empty())
      return DistributedValue::voidReturnValue();

    if (HFA.has_value()) {
      // Every field is returned in a register of its own.
      Limit = HFA->FieldCount;

    } else {
      // TODO: replace this the explicit single register limit with
      //       an abi-defined value. For more information see the relevant
      //       comment in `distributeRegisterArguments`.
      Limit = 1;
    }

  } else if (Strategy == FPS::NextRegisterFromTheGeneralPurposeRegisterSet) {
    RegisterList = ABI.GeneralPurposeReturnValueRegisters();
    Limit = T.isScalar() ? ABI.MaximumGPRsPerScalarReturnValue() :
                           ABI.MaximumGPRsPerAggregateReturnValue();

  } else {
    revng_abort("Unknown floating point strategy.");
  }

  RegisterPortions Portions = fullPortions(RegisterList);
  if (HFA.has_value())
    for (model::Register::Portion &Portion : Portions)
      Portion.Size = HFA->FieldSize;

  auto &&[Result, _] = distribute(T, Portions, 0, Limit, true);
  revng_assert(Result.size() == 1, "Return values should never be padded.");
  return Result[0];
}

} // namespace abi::FunctionType
