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
  revng_log(Log, "Original Type:\n" << serializeToString(FunctionType));
  if (auto *StackType = FunctionType.stackArgumentsType())
    revng_log(Log, "Stack is:\n" << serializeToString(*StackType));
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
  auto [Definition, Type] = Binary->makeCABIFunctionDefinition();
  revng_assert(Definition.ID() != 0);
  model::copyMetadata(Definition, FunctionType);
  Definition.ABI() = ABI.ABI();

  // And filled in with the argument information.
  auto Arguments = llvm::concat<model::Argument>(Converted->RegisterArguments,
                                                 Converted->StackArguments);
  for (auto &Argument : Arguments)
    Definition.Arguments().insert(std::move(Argument));
  Definition.ReturnType() = Converted->ReturnValueType;

  revng_log(Log, "Conversion successful:\n" << serializeToString(Definition));

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

static bool verifyAlignment(const abi::Definition &ABI,
                            uint64_t CurrentOffset,
                            uint64_t CurrentSize,
                            uint64_t NextOffset,
                            uint64_t NextAlignment) {
  uint64_t PaddedSize = ABI.paddedSizeOnStack(CurrentSize);
  revng_log(Log,
            "Attempting to verify alignment for an argument with size "
              << CurrentSize << " (" << PaddedSize << " when padded) at offset "
              << CurrentOffset << ". The next argument is at offset "
              << NextOffset << " and is aligned at " << NextAlignment << ".");

  OverflowSafeInt Offset = CurrentOffset;
  Offset += PaddedSize;
  if (!Offset) {
    // Abandon if offset overflows.
    revng_log(Log, "Error: Integer overflow when calculating field offsets.");
    return false;
  }

  if (*Offset == NextOffset) {
    revng_log(Log, "Argument slots in perfectly.");
    return true;
  } else if (*Offset < NextOffset) {
    // Offsets are different, there's most likely padding between the arguments.
    revng_assert(NextOffset % NextAlignment == 0,
                 "Type's alignment doesn't make sense.");
    uint64_t AdjustedAlignment = NextAlignment;

    // Round all the alignment up to the register size - to avoid sub-word
    // offsets on the stack.
    if (AdjustedAlignment < ABI.getPointerSize())
      AdjustedAlignment = ABI.getPointerSize();
    if (NextOffset % AdjustedAlignment != 0) {
      std::string Error = "Adjusted alignment of "
                          + std::to_string(AdjustedAlignment)
                          + " (next alignment is "
                          + std::to_string(NextAlignment)
                          + ") doesn't make any sense for an offset of "
                          + std::to_string(NextOffset) + ".";
      revng_abort(Error.c_str());
    }

    // Check whether the next argument's position makes sense.
    uint64_t Delta = AdjustedAlignment - *Offset % AdjustedAlignment;
    if (Delta != AdjustedAlignment && *Offset + Delta == NextOffset) {
      revng_log(Log,
                "Argument slots in after accounting for the alignment of the "
                "next field adjusted to "
                  << AdjustedAlignment << " with the resulting difference of "
                  << Delta << ".");
      return true;
    } else {
      revng_log(Log,
                "Error: The natural alignment of a type would make it "
                "impossible to represent as CABI: there would have to be "
                "a hole between two arguments. Abandon the conversion.");
      // TODO: we probably want to preprocess such functions and manually
      //       "fill" the holes in before attempting the conversion.
      return false;
    }
  } else {
    revng_log(Log,
              "Error: The natural alignment of a type would make it "
              "impossible to represent as CABI: the arguments (including "
              "the padding) would have to overlap. Abandon the conversion.");
    return false;
  }
}

template<model::AnyType ModelType>
bool canBeNext(ArgumentDistributor &Distributor,
               const ModelType &CurrentType,
               uint64_t CurrentOffset,
               uint64_t NextOffset,
               uint64_t NextAlignment) {
  const abi::Definition &ABI = Distributor.ABI;

  revng_log(Log,
            "Checking whether the argument #"
              << Distributor.ArgumentIndex << " can be slotted right in.\n"
              << "Currently " << Distributor.UsedGeneralPurposeRegisterCount
              << " general purpose and " << Distributor.UsedVectorRegisterCount
              << " vector registers are in use.");

  std::optional<uint64_t> Size = CurrentType.size();
  revng_assert(Size.has_value() && Size.value() != 0);

  if (!verifyAlignment(ABI, CurrentOffset, *Size, NextOffset, NextAlignment))
    return false;

  for (const auto &Distributed : Distributor.nextArgument(CurrentType)) {
    if (Distributed.RepresentsPadding)
      continue; // Skip padding.

    if (!Distributed.Registers.empty()) {
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
    uint64_t NextStackOffset = ABI.alignedOffset(Distributor.UsedStackOffset,
                                                 CurrentType);
    NextStackOffset += ABI.paddedSizeOnStack(*Size);
    uint64_t SizeWithPadding = NextStackOffset - Distributor.UsedStackOffset;
    if (Distributed.SizeOnStack != SizeWithPadding) {
      revng_log(Log,
                "Because the stack position wouldn't match due to holes in "
                "the stack, an argument cannot just be added as is - resulting "
                "CFT would not be compatible. Expected size is "
                  << SizeWithPadding << " but distributor reports "
                  << Distributed.SizeOnStack << " instead.");
      return false;
    }
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

  // As a workaround for the cases where expected alignment is missing at
  // the very end of the RFT-style stack argument struct, use adjusted
  // size and alignment values instead of the real ones.
  auto &Stack = StackStruct->toStruct();
  uint64_t StackAlignment = *ABI.alignment(Stack);
  revng_assert(llvm::isPowerOf2_64(StackAlignment));
  uint64_t AdjustedAlignment = std::max(StackAlignment, ABI.getPointerSize());
  uint64_t StackSize = abi::FunctionType::paddedSizeOnStack(Stack.Size(),
                                                            AdjustedAlignment);

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

    // Verify the alignment of the first argument.
    if (Stack.Fields().empty()) {
      revng_log(Log, "Stack struct has no fields.");
    } else {
      uint64_t FirstAlignment = *ABI.alignment(*Stack.Fields().begin()->Type());
      revng_assert(llvm::isPowerOf2_64(FirstAlignment));
    }

    // Go over all the `[CurrentArgument, TheNextOne]` field pairs.
    // `CurrentRange` is used to keep track of the "remaining" ones.
    // This allows us to keep its value when the loop to be aborted (which
    // happens when we meet an argument we cannot just convert "as is").
    auto CurrentRange = std::ranges::subrange(Stack.Fields().begin(),
                                              Stack.Fields().end());
    bool NaiveConversionFailed = CurrentRange.empty();
    while (CurrentRange.size() > 1) {
      auto [CurrentArgument, TheNextOne] = takeAsTuple<2>(CurrentRange);

      uint64_t NextAlignment = *ABI.alignment(*TheNextOne.Type());
      revng_assert(llvm::isPowerOf2_64(NextAlignment));

      if (!canBeNext(Distributor,
                     *CurrentArgument.Type(),
                     CurrentArgument.Offset(),
                     TheNextOne.Offset(),
                     NextAlignment)) {
        // We met a argument we cannot just "add" as is.
        NaiveConversionFailed = true;
        break;
      }

      // Create the argument from this field.
      model::Argument &New = Result.emplace_back();
      model::copyMetadata(New, CurrentArgument);
      New.Index() = Distributor.ArgumentIndex - 1;
      New.Type() = CurrentArgument.Type();

      CurrentRange = std::ranges::subrange(std::next(CurrentRange.begin()),
                                           Stack.Fields().end());
    }

    // The main loop is over. Which means that we either converted everything
    // but the very last argument correctly, OR that we aborted half-way
    // through.
    if (CurrentRange.size() == 1) {
      revng_assert(NaiveConversionFailed == false);

      // Having only one element in the "remaining" range means that only
      // the last field is left - add it too after checking.
      const model::StructField &LastArgument = CurrentRange.front();
      std::optional<uint64_t> LastSize = LastArgument.Type()->size();
      revng_assert(LastSize.has_value() && LastSize.value() != 0);
      if (canBeNext(Distributor,
                    *LastArgument.Type(),
                    LastArgument.Offset(),
                    StackSize,
                    StackAlignment)) {
        model::Argument &New = Result.emplace_back();
        model::copyMetadata(New, LastArgument);
        New.Type() = LastArgument.Type();
        New.Index() = Distributor.ArgumentIndex - 1;
        return Result;
      }

      NaiveConversionFailed = true;
    }

    // Getting to this point (past the return statement in the last element
    // section) means that there is at least one argument we cannot just "add".
    //
    // TODO: consider using more robust approaches here, maybe an attempt to
    //       re-start adding argument normally after some "nonconforming"
    //       structs are added.
    revng_assert(NaiveConversionFailed == true);
    revng_log(Log,
              "Naive conversion failed. Try to fall back on using structs "
              "instead.");
    if (CurrentRange.size() != Stack.Fields().size()) {
      // This condition being true means that we did succeed in converting some
      // of the arguments, but failed on some others. Let's try to wrap
      // the remainder into a struct and see if bundled together they make more
      // sense in c-like representation.
      revng_log(Log,
                "Some fields were converted successfully, try to slot in the "
                "rest as a struct.");
      const model::StructField &LastSuccess = *std::prev(CurrentRange.begin());
      std::uint64_t Offset = LastSuccess.Offset();
      Offset += ABI.paddedSizeOnStack(*LastSuccess.Type()->size());

      model::StructDefinition RemainingArguments;
      RemainingArguments.Size() = Stack.Size() - Offset;
      for (const auto &Field : CurrentRange) {
        model::StructField Copy = Field;
        Copy.Offset() -= Offset;
        RemainingArguments.Fields().emplace(std::move(Copy));
      }

      auto &RA = RemainingArguments;
      if (canBeNext(Distributor, RA, Offset, Stack.Size(), StackAlignment)) {
        revng_log(Log, "Struct for the remaining argument worked.");
        model::Argument &New = Result.emplace_back();
        New.Index() = Stack.Fields().size() - CurrentRange.size();
        New.Index() += InitialIndexOffset;

        New.Type() = Bucket.makeStructDefinition(std::move(RA)).second;
        return Result;
      }
    }
  }

  // Reaching this far means that we either aborted on the very first argument
  // OR that the partial conversion didn't work well either OR that stack struct
  // has explicit metadata (i.e. name, comments, etc) so we don't want to break
  // it apart.
  // Let's try and see if it would make sense to add the whole "stack" struct
  // as one argument.
  if (!canBeNext(Distributor, *StackStruct, 0, StackSize, StackAlignment)) {
    // Nope, stack struct didn't work either. There's nothing else we can do.
    // Just report that this function cannot be converted.
    return std::nullopt;
  }

  revng_log(Log,
            "Adding the whole stack as a single argument is the best we can "
            "do.");
  model::Argument &New = Result.emplace_back();
  New.Type() = StackStruct.copy();
  New.Index() = InitialIndexOffset;

  return Result;
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
      auto [Definition, ReturnType] = Bucket.makeStructDefinition();
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
        uint64_t RegisterSize = model::ABI::getPointerSize(ABI.ABI());
        Definition.Size() += paddedSizeOnStack(FieldSize.value(), RegisterSize);
      }

      revng_assert(Definition.Size() != 0 && !Definition.Fields().empty());
      return std::move(ReturnType);
    }
  }
}

} // namespace abi::FunctionType
