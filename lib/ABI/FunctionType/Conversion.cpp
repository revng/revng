/// \file Conversion.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ADT/STLExtras.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/TypeBucket.h"
#include "revng/Support/OverflowSafeInt.h"

#include "ValueDistributor.h"

static Logger Log("function-type-conversion-to-cabi");

namespace abi::FunctionType {

class ToCABIConverter {
private:
  using RFT = const model::RawFunctionDefinition &;
  using RFTArguments = decltype(std::declval<RFT>().Arguments());
  using RFTReturnValues = decltype(std::declval<RFT>().ReturnValues());

public:
  struct Converted {
    llvm::SmallVector<model::Argument, 8> RegisterArguments;
    llvm::SmallVector<model::Argument, 8> StackArguments;
    model::UpcastableType ReturnValueType;
  };

private:
  const abi::Definition &ABI;
  model::TypeBucket Bucket;
  const bool UseSoftRegisterStateDeductions = false;

  abi::Definition::AlignmentCache AlignmentCache = {};

public:
  ToCABIConverter(const abi::Definition &ABI,
                  model::Binary &Binary,
                  const bool UseSoftRegisterStateDeductions) :
    ABI(ABI),
    Bucket(Binary),
    UseSoftRegisterStateDeductions(UseSoftRegisterStateDeductions) {}

  [[nodiscard]] std::optional<Converted>
  tryConvert(const model::RawFunctionDefinition &FunctionType) {
    // Register arguments first.
    auto Arguments = tryConvertingRegisterArguments(FunctionType.Arguments());
    if (!Arguments.has_value()) {
      revng_log(Log, "Unable to convert register arguments.");
      Bucket.drop();
      return std::nullopt;
    }

    // Count used registers.
    ArgumentDistributor Distributor(ABI);
    Distributor.ArgumentIndex = Arguments->size();
    for (const auto &NTRegister : FunctionType.Arguments()) {
      auto Kind = model::Register::primitiveKind(NTRegister.Location());
      if (Kind == model::PrimitiveKind::Values::PointerOrNumber)
        ++Distributor.UsedGeneralPurposeRegisterCount;
      else if (Kind == model::PrimitiveKind::Float)
        ++Distributor.UsedVectorRegisterCount;
      else
        revng_abort(("Register ("
                     + model::Register::getName(NTRegister.Location()).str()
                     + ") is not supported as an argument.")
                      .c_str());
    }

    // Then stack ones.
    auto Stack = tryConvertingStackArguments(FunctionType.StackArgumentsType(),
                                             Distributor);
    if (!Stack.has_value()) {
      revng_log(Log, "Unable to convert stack arguments.");
      Bucket.drop();
      return std::nullopt;
    }

    // And the return value.
    auto ReturnType = tryConvertingReturnValue(FunctionType.ReturnValues());
    if (!ReturnType.has_value()) {
      revng_log(Log, "Unable to convert return value.");
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
  tryConvertingRegisterArguments(RFTArguments Registers);

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
  tryConvertingStackArguments(const model::UpcastableType &StackStruct,
                              ArgumentDistributor Distributor);

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
  std::optional<model::UpcastableType>
  tryConvertingReturnValue(RFTReturnValues Registers);

private:
  template<model::AnyType AnyType>
  uint64_t adjustedAlignment(const AnyType &Type) {
    std::optional IsNatural = ABI.hasNaturalAlignment(Type, AlignmentCache);
    revng_assert(IsNatural.has_value());
    if (not *IsNatural)
      return ABI.MinimumStackArgumentSize();

    std::optional<uint64_t> Result = *ABI.alignment(Type, AlignmentCache);
    revng_assert(Result.has_value());
    if (ABI.PackStackArguments())
      return *Result;

    return std::max(*Result, ABI.MinimumStackArgumentSize());
  }
};

std::optional<model::UpcastableType>
tryConvertToCABI(const model::RawFunctionDefinition &FunctionType,
                 TupleTree<model::Binary> &Binary,
                 std::optional<model::ABI::Values> MaybeABI,
                 bool UseSoftRegisterStateDeductions) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();

  revng_log(Log,
            "Converting a `RawFunctionDefinition` to "
            "`CABIFunctionDefinition`.");
  revng_log(Log, "ABI: " << model::ABI::getName(MaybeABI.value()).str());
  revng_log(Log, "Original Type:\n" << toString(FunctionType));
  if (auto *StackType = FunctionType.stackArgumentsType())
    revng_log(Log, "Stack is:\n" << toString(*StackType));
  LoggerIndent Indentation(Log);

  const abi::Definition &ABI = abi::Definition::get(*MaybeABI);
  if (!ABI.isPreliminarilyCompatibleWith(FunctionType)) {
    revng_log(Log,
              "FAIL: the function is not compatible with `"
                << model::ABI::getName(ABI.ABI()) << "`.");
    return std::nullopt;
  }

  ToCABIConverter Converter(ABI, *Binary, UseSoftRegisterStateDeductions);
  std::optional Converted = Converter.tryConvert(FunctionType);
  if (!Converted.has_value())
    return std::nullopt;

  // The conversion was successful, a new `CABIFunctionDefinition` can now be
  // created,
  auto &&[Definition, Type] = Binary->makeCABIFunctionDefinition();
  revng_assert(Definition.ID() != 0);
  model::copyMetadata(Definition, FunctionType);
  Definition.ABI() = ABI.ABI();

  // And filled in with the argument information.
  auto Arguments = llvm::concat<model::Argument>(Converted->RegisterArguments,
                                                 Converted->StackArguments);
  for (auto &Argument : Arguments)
    Definition.Arguments().insert(std::move(Argument));
  Definition.ReturnType() = Converted->ReturnValueType;

  revng_log(Log, "Conversion successful:\n" << toString(Definition));

  // Since CABI-FT only have one field for return value comments - we have no
  // choice but to resort to concatenation in order to preserve as much
  // information as possible.
  for (model::NamedTypedRegister ReturnValue : FunctionType.ReturnValues()) {
    if (!ReturnValue.Comment().empty()) {
      if (!Definition.ReturnValueComment().empty())
        Definition.ReturnValueComment() += '\n';
      model::Register::Values RVL = ReturnValue.Location();
      Definition.ReturnValueComment() += model::Register::getRegisterName(RVL);
      Definition.ReturnValueComment() += ": ";
      Definition.ReturnValueComment() += ReturnValue.Comment();
    }
  }

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceTypeDefinition(FunctionType.key(), *Type, Binary);

  // And don't forget to remove the old type.
  Binary->TypeDefinitions().erase(FunctionType.key());

  return std::move(Type);
}

using TCC = ToCABIConverter;
std::optional<llvm::SmallVector<model::Argument, 8>>
TCC::tryConvertingRegisterArguments(RFTArguments Registers) {
  // Rely onto the register state deduction to make sure no "holes" are
  // present in-between the argument registers.
  auto Unwrap = std::views::transform([](const model::NamedTypedRegister &R) {
    return R.Location();
  });
  auto Deduced = Registers | Unwrap | revng::to<abi::Definition::RegisterSet>();

  if (!UseSoftRegisterStateDeductions)
    Deduced = ABI.enforceArgumentRegisterState(std::move(Deduced));
  else if (auto R = ABI.tryDeducingArgumentRegisterState(std::move(Deduced)))
    Deduced = *R;
  else
    return std::nullopt;

  // But just knowing which registers we need is not sufficient, we also have to
  // order them properly.
  auto Ordered = ABI.sortArguments(std::move(Deduced));

  llvm::SmallVector<model::Argument, 8> Result;
  for (model::Register::Values Register : Ordered) {
    auto CurrentRegisterIterator = Registers.find(Register);
    if (CurrentRegisterIterator != Registers.end()) {
      // If the current register is confirmed to be in use, convert it into
      // an argument while preserving its type and metadata.
      model::Argument &Current = Result.emplace_back();
      Current.CustomName() = CurrentRegisterIterator->CustomName();

      revng_assert(!CurrentRegisterIterator->Type().isEmpty());
      Current.Type() = CurrentRegisterIterator->Type().copy();
    } else {
      // This register is unused but we still have to create an argument
      // for it. Otherwise the resulting function will be semantically different
      // from the input one.
      //
      // Also, if the indicator for this "hole" is not preserved, there will be
      // no way to recreate it at any point in the future, when it's being
      // converted back to the representation similar to original (e.g. when
      // producing the `Layout` for this function).
      using Primitive = model::PrimitiveType;
      Result.emplace_back().Type() = Primitive::makeGeneric(Register);
    }
  }

  // Rewrite all the indices to make sure they are consistent.
  for (auto Pair : llvm::enumerate(Result))
    Pair.value().Index() = Pair.index();

  return Result;
}

struct ArgumentProperties {
  uint64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t Alignment = 0;
};

/// \note: if `Previous` is `std::nullopt`, it means this is the first argument.
static bool verifyAlignment(const abi::Definition &ABI,
                            const ArgumentProperties &Current,
                            const std::optional<ArgumentProperties> &Previous) {
  revng_log(Log,
            "Attempting to verify alignment for an argument with size "
              << Current.Size << " (aligned at " << Current.Alignment
              << " bytes) at offset " << Current.Offset << ".");
  if (Previous.has_value()) {
    revng_log(Log,
              "The previous argument is "
                << Previous->Size << " bytes (aligned at "
                << Previous->Alignment << " bytes) at offset "
                << Previous->Offset << ".");
  } else {
    revng_log(Log, "This is the first non-register argument.");
  }

  if (Current.Offset % Current.Alignment != 0) {
    revng_log(Log, "Error: Argument is not correctly aligned.");
    return false;
  }

  uint64_t ExpectedOffset = 0;
  if (Previous.has_value()) {
    uint64_t PaddedSize = paddedSizeOnStack(Previous->Size,
                                            Previous->Alignment);

    if (PaddedSize != Previous->Size)
      revng_log(Log,
                "The previous argument had to be padded to "
                  << PaddedSize << " bytes because of alignment constraints.");

    OverflowSafeInt OverflowCheck = Previous->Offset;
    OverflowCheck += PaddedSize;
    if (!OverflowCheck) {
      // Abandon if offset overflows.
      revng_log(Log, "Error: Integer overflow when calculating field offsets.");
      return false;
    }

    ExpectedOffset = *OverflowCheck;
  }

  ExpectedOffset = ABI.alignedOffset(ExpectedOffset, Current.Alignment);

  if (ExpectedOffset == Current.Offset) {
    revng_log(Log, "Argument slots in perfectly.");
    return true;

  } else if (ExpectedOffset < Current.Offset) {
    revng_log(Log,
              "Error: The natural alignment of a type would make it "
              "impossible to represent as CABI:\n"
              "There would have to be a hole between two arguments.");

    // TODO: we probably want to preprocess such functions and manually
    //       "fill" the holes in before attempting the conversion.
    return false;

  } else {
    revng_log(Log,
              "Error: The natural alignment of a type would make it "
              "impossible to represent as CABI:\n"
              "The arguments (including the padding) would have to overlap.");
    return false;
  }
}

template<model::AnyType ModelType>
bool canBeNext(ArgumentDistributor &Distributor,
               const ModelType &CurrentType,
               const ArgumentProperties &Current,
               const std::optional<ArgumentProperties> &Previous) {
  const abi::Definition &ABI = Distributor.ABI;

  revng_log(Log,
            "Checking whether the argument #"
              << Distributor.ArgumentIndex << " can be slotted right in.\n"
              << "Currently " << Distributor.UsedGeneralPurposeRegisterCount
              << " general purpose and " << Distributor.UsedVectorRegisterCount
              << " vector registers are in use.");

  if (!verifyAlignment(ABI, Current, Previous))
    return false;

  // We want to ensure the distributor does not change on a failed argument.
  // The easiest way to do that is to work on a copy.
  ArgumentDistributor LocalDistributor = Distributor;

  for (const auto &Distributed : LocalDistributor.nextArgument(CurrentType)) {
    if (Distributed.RepresentsPadding)
      continue; // Skip padding.

    if (!Distributed.Registers.empty()) {
      // TODO: we might want to consider filling such holes manually, either
      //       here or in a different analysis.
      revng_log(Log,
                "Error: Because there are still available registers, "
                "an argument cannot just be added as is - resulting CFT would "
                "not be compatible with the original function.");
      return false;
    }
    if (Distributed.SizeOnStack == 0) {
      revng_log(Log,
                "Something went very wrong: an argument uses neither registers "
                "nor stack.");
      return false;
    }

    // Compute the next stack offset
    auto NextStackOffset = ABI.alignedOffset(LocalDistributor.UsedStackOffset,
                                             CurrentType);
    NextStackOffset += ABI.paddedSizeOnStack(Current.Size);
    auto SizeWithPadding = NextStackOffset - LocalDistributor.UsedStackOffset;
    if (Distributed.SizeOnStack != SizeWithPadding) {
      revng_log(Log,
                "Because the stack position wouldn't match due to holes in "
                "the stack, an argument cannot just be added as is - resulting "
                "CFT would not be compatible. Expected size is "
                  << SizeWithPadding << " but distributor reports "
                  << Distributed.SizeOnStack << " instead.");
      return false;
    }

    // The distribution succeeded, apply the distributor
    std::swap(LocalDistributor, Distributor);
  }

  return true;
}

std::optional<llvm::SmallVector<model::Argument, 8>>
TCC::tryConvertingStackArguments(const model::UpcastableType &StackStruct,
                                 ArgumentDistributor Distributor) {
  if (StackStruct.isEmpty()) {
    // If there is no type, it means that the importer responsible for this
    // function didn't detect any stack arguments and avoided creating
    // a new empty type.
    return llvm::SmallVector<model::Argument, 8>{};
  }
  auto &Stack = StackStruct->toStruct();
  uint64_t StackSize = paddedSizeOnStack(Stack.Size(),
                                         adjustedAlignment(Stack));

  // If the struct is empty, it indicates that there are no stack arguments.
  if (Stack.Size() == 0) {
    revng_assert(Stack.Fields().empty());
    return llvm::SmallVector<model::Argument, 8>{};
  }

  llvm::SmallVector<model::Argument, 8> Result;
  uint64_t InitialIndexOffset = Distributor.ArgumentIndex;
  if (!StackStruct->isTypedef() && !model::hasMetadata(Stack)) {
    // NOTE: Only proceed with trying to "split" the stack struct up if it has
    //       no metadata attached: as in, it doesn't have a user-defined
    //       `CustomName` or a `Comment` or any other fields we'll mark as
    //       metadata in the future.

    if (Stack.Fields().empty())
      revng_log(Log, "Stack struct has no fields.");

    // If none of our unrolling attempts work and we have to fall back onto
    // the full struct distribution, we'd need a "fresh" distributor, so make
    // a copy of one now.
    ArgumentDistributor BackupDistributor = Distributor;

    std::optional<ArgumentProperties> PreviousArgumentProperties = std::nullopt;

    // Keeping this as a remaining argument range allows us to keep its value
    // even when the loop is aborted (which happens when we meet an argument
    // we cannot just convert "as is").
    auto RemainingRange = std::ranges::subrange(Stack.Fields().begin(),
                                                Stack.Fields().end());

    // Some ABIs require the stack arguments to start at an offset (usually 4
    // words, enough space to put all the register arguments). This branch
    // handles that by adjusting the `RemainingRange` to account for it.
    if (auto SkippedBytes = ABI.UnusedStackArgumentBytes()) {
      revng_assert(SkippedBytes % ABI.getPointerSize() == 0);

      auto Iterator = Stack.Fields().find(SkippedBytes);
      if (Iterator == Stack.Fields().end()) {
        revng_log(Log,
                  "Area at the top of the stack dedicated for storing "
                  "registers seems malformed (as in, there is no field for "
                  "the *actual* first stack argument), abort the conversion.");
        return std::nullopt;
      }

      revng_log(Log,
                "Skipping some fields because they fall into an reserved part "
                "of the stack.");

      RemainingRange = std::ranges::subrange(Iterator, Stack.Fields().end());

      if (Iterator != Stack.Fields().begin()) {
        auto Previous = std::prev(Iterator);
        PreviousArgumentProperties = ArgumentProperties{
          .Offset = Previous->Offset(),
          .Size = *Previous->Type()->size(),
          .Alignment = adjustedAlignment(*Previous->Type())
        };
      }
    }

    while (not RemainingRange.empty()) {
      const auto &CurrentArgument = *RemainingRange.begin();

      ArgumentProperties CurrentArgumentProperties = {
        .Offset = CurrentArgument.Offset(),
        .Size = *CurrentArgument.Type()->size(),
        .Alignment = adjustedAlignment(*CurrentArgument.Type())
      };

      if (!canBeNext(Distributor,
                     *CurrentArgument.Type(),
                     CurrentArgumentProperties,
                     PreviousArgumentProperties)) {
        // We met a argument we cannot just "add" as is.
        break;
      }

      // Create the argument from this field.
      model::Argument &New = Result.emplace_back();
      model::copyMetadata(New, CurrentArgument);
      New.Index() = Distributor.ArgumentIndex - 1;
      New.Type() = CurrentArgument.Type();

      PreviousArgumentProperties = CurrentArgumentProperties;
      RemainingRange = std::ranges::subrange(std::next(RemainingRange.begin()),
                                             Stack.Fields().end());
    }

    if (PreviousArgumentProperties.has_value()) {
      // Having previous argument property set means that we successfully
      // distributed at least one argument, but it doesn't mean we have handled
      // all of them.
      //
      // So that's exactly what we need to check for now.

      auto [PreviousOffset,
            PreviousSize,
            PreviousAlignment] = *PreviousArgumentProperties;
      uint64_t StartingOffset = PreviousOffset
                                + paddedSizeOnStack(PreviousSize,
                                                    PreviousAlignment);

      uint64_t CurrentAlignment = ABI.MinimumStackArgumentSize();
      if (not RemainingRange.empty())
        CurrentAlignment = adjustedAlignment(*RemainingRange.begin()->Type());
      StartingOffset = ABI.alignedOffset(StartingOffset, CurrentAlignment);

      if (StartingOffset < Stack.Size()) {
        // We still have unhandled data at the end of the stack,
        // try to pack them into a struct and see if such a struct would
        // make any sense.
        revng_log(Log,
                  "Some fields were converted successfully, try to slot in the "
                  "rest as a struct.");

        if (RemainingRange.empty()
            or RemainingRange.begin()->Offset() >= StartingOffset) {

          revng_log(Log,
                    "Such a struct would start on a misaligned offset, "
                    "which is not supported.");

          model::StructDefinition RemainingArgs;
          RemainingArgs.Size() = Stack.Size() - StartingOffset;
          for (const auto &Field : RemainingRange) {
            model::StructField Copy = Field;
            revng_assert(Copy.Offset() >= StartingOffset);
            Copy.Offset() -= StartingOffset;
            RemainingArgs.Fields().emplace(std::move(Copy));
          }

          ArgumentProperties RemainingProperties = {
            .Offset = StartingOffset,
            .Size = RemainingArgs.Size(),
            .Alignment = adjustedAlignment(RemainingArgs)
          };

          const auto &PAS = PreviousArgumentProperties;
          if (canBeNext(Distributor, RemainingArgs, RemainingProperties, PAS)) {
            revng_log(Log, "Struct for the remaining arguments worked.");
            model::Argument &New = Result.emplace_back();
            New.Index() = Distributor.ArgumentIndex - 1;

            auto &RA = RemainingArgs;
            New.Type() = Bucket.makeStructDefinition(std::move(RA)).second;

            return Result;
          }
        }
      } else {
        if (StartingOffset != Stack.Size()) {
          revng_log(Log,
                    "WARNING: The struct ending point does not match the "
                    "expected alignment ("
                      << StartingOffset << " is expected, but " << Stack.Size()
                      << " was found).\n"
                      << "Note that this is not necessarily a problem on its "
                         "own, but it might be an indication that we're "
                         "mishandling something, so proceed with caution.");
        }

        // All good, all the arguments make sense!
        return Result;
      }
    }

    // Reaching this far means that we couldn't do much and that our best bet is
    // to try and use the entire stack struct as a single argument.
    //
    // Note that any distribution attempts we made before this should be
    // discarded before we proceed. And the easiest way to do that is to restore
    // the distributor from the backup created earlier.
    std::swap(BackupDistributor, Distributor);
    Result = {};
  }

  if (ABI.UnusedStackArgumentBytes()) {
    revng_log(Log,
              "Attaching the whole stack as a single struct is not supported "
              "under the ABIs that require a part of it to be stripped "
              "(see `UnusedStackArgumentBytes` within a given ABI definition "
              "for more information).");
    // TODO: We might be able to strip just the problem bytes from the top,
    //       but abort the entire conversion for now.
    return std::nullopt;
  }

  ArgumentProperties FullStackProperties = {
    .Offset = 0, .Size = Stack.Size(), .Alignment = adjustedAlignment(Stack)
  };

  if (!canBeNext(Distributor,
                 *StackStruct,
                 FullStackProperties,
                 std::nullopt)) {
    // Nope, stack struct didn't work either. There's nothing else we can do.
    // Just report that this function cannot be converted.
    return std::nullopt;
  }

  revng_log(Log,
            "Adding the whole stack as a single argument is the best we can "
            "do.");
  model::Argument New;
  New.Type() = StackStruct.copy();
  New.Index() = InitialIndexOffset;

  return { { New } };
}

std::optional<model::UpcastableType>
TCC::tryConvertingReturnValue(RFTReturnValues Registers) {
  if (Registers.size() == 0) {
    // The function doesn't return anything.
    return model::UpcastableType{};
  }

  // We only convert register-based return values: those that are returned using
  // a pointer to memory readied by the callee are technically fine without any
  // intervention (they are just `void` functions that modify some object passed
  // into them with a pointer.
  //
  // We might want to handle this in a different way under some architectures
  // (i.e. ARM64 because it uses a separate `PointerToCopyLocation` register),
  // but for now the dumb approach should suffice.

  auto Unwrap = std::views::transform([](const model::NamedTypedRegister &R) {
    return R.Location();
  });
  auto Deduced = Registers | Unwrap | revng::to<abi::Definition::RegisterSet>();

  if (!UseSoftRegisterStateDeductions)
    Deduced = ABI.enforceReturnValueRegisterState(std::move(Deduced));
  else if (auto R = ABI.tryDeducingReturnValueRegisterState(std::move(Deduced)))
    Deduced = *R;
  else
    return std::nullopt;

  auto Ordered = ABI.sortReturnValues(std::move(Deduced));

  if (Ordered.size() == 1) {
    if (auto Iter = Registers.find(*Ordered.begin()); Iter != Registers.end()) {
      // Only one register is used, just return its type.
      return Iter->Type();
    } else {
      // One register is used but its type cannot be obtained.
      // Create a register sized return type instead.
      return model::PrimitiveType::make(*Ordered.begin());
    }
  } else {
    // Multiple registers, it's either a struct or a big scalar.

    // First, check if it can possibly be a struct.
    if (ABI.MaximumGPRsPerAggregateReturnValue() < Ordered.size()) {
      // It cannot be a struct, it's bigger than allowed.
      if (ABI.MaximumGPRsPerScalarReturnValue() < Ordered.size()) {
        // It cannot be a scalar either.
        revng_log(Log,
                  "No known return value type supports that many registers ("
                    << Ordered.size() << ") under "
                    << model::ABI::getName(ABI.ABI()) << ":\n");
        return std::nullopt;
      }

      // It's probably a scalar, replace its type with a fake one making sure
      // at least the size adds up.
      //
      // TODO: sadly this discards type information from the registers, look
      //       into preserving it at least partially.
      uint64_t PointerSize = model::ABI::getPointerSize(ABI.ABI());
      uint64_t PrimitiveSize = Ordered.size() * PointerSize;
      if (llvm::is_contained(ABI.ScalarTypes(), PrimitiveSize)) {
        return model::PrimitiveType::makeGeneric(PrimitiveSize);
      } else {
        revng_log(Log,
                  "The primitive return value ("
                    << Ordered.size() << " bytes) is not a valid scalar under "
                    << model::ABI::getName(ABI.ABI()) << ":\n");
        return std::nullopt;
      }
    } else {
      // It could be either a struct or a scalar, go the conservative route
      // and make a struct for it.
      auto &&[Definition, ReturnType] = Bucket.makeStructDefinition();
      for (model::Register::Values Register : Ordered) {
        // Make a separate field for each register.
        model::StructField Field;
        Field.Offset() = Definition.Size();

        // TODO: ensure that the type is in fact naturally aligned
        if (auto It = Registers.find(*Ordered.begin()); It != Registers.end())
          Field.Type() = It->Type();
        else
          Field.Type() = model::PrimitiveType::makeGeneric(*Ordered.begin());

        std::optional<uint64_t> FieldSize = Field.Type()->size();
        revng_assert(FieldSize.has_value() && FieldSize.value() != 0);

        // Round the next offset based on the natural alignment.
        Definition.Size() = ABI.alignedOffset(Definition.Size(), *Field.Type());

        // Insert the field
        Definition.Fields().insert(std::move(Field));

        // Update the total struct size: insert some padding if necessary.
        Definition.Size() += paddedSizeOnStack(FieldSize.value(),
                                               ABI.MinimumStackArgumentSize());
      }

      revng_assert(Definition.Size() != 0 && !Definition.Fields().empty());
      return std::move(ReturnType);
    }
  }
}

} // namespace abi::FunctionType
