/// \file Conversion.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallSet.h"

#include "revng/ABI/Definition.h"
#include "revng/ABI/FunctionType/Conversion.h"
#include "revng/ABI/FunctionType/Support.h"
#include "revng/ABI/FunctionType/TypeBucket.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
#include "revng/Support/OverflowSafeInt.h"

static Logger Log("function-type-conversion-to-cabi");

namespace abi::FunctionType {

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

private:
  const abi::Definition &ABI;
  TypeBucket Bucket;
  const bool UseSoftRegisterStateDeductions;

public:
  ToCABIConverter(const abi::Definition &ABI,
                  model::Binary &Binary,
                  const bool UseSoftRegisterStateDeductions) :
    ABI(ABI),
    Bucket(Binary),
    UseSoftRegisterStateDeductions(UseSoftRegisterStateDeductions) {}

  [[nodiscard]] std::optional<Converted>
  tryConvert(const model::RawFunctionType &FunctionType) {
    // Register arguments first.
    auto Arguments = tryConvertingRegisterArguments(FunctionType.Arguments());
    if (!Arguments.has_value()) {
      revng_log(Log, "Unable to convert register arguments.");
      Bucket.drop();
      return std::nullopt;
    }

    // Then stack ones.
    auto Stack = tryConvertingStackArguments(FunctionType.StackArgumentsType(),
                                             Arguments->size());
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
                 std::optional<model::ABI::Values> MaybeABI,
                 bool UseSoftRegisterStateDeductions) {
  if (!MaybeABI.has_value())
    MaybeABI = Binary->DefaultABI();

  revng_log(Log, "Converting a `RawFunctionType` to `CABIFunctionType`.");
  revng_log(Log, "ABI: " << model::ABI::getName(MaybeABI.value()).str());
  revng_log(Log, "Original Type:\n" << serializeToString(FunctionType));
  LoggerIndent Indentation(Log);

  const abi::Definition &ABI = abi::Definition::get(*MaybeABI);
  if (ABI.isIncompatibleWith(FunctionType)) {
    revng_log(Log,
              "FAIL: the function is not compatible with `"
                << model::ABI::getName(ABI.ABI()) << "`.");
    return std::nullopt;
  }

  ToCABIConverter Converter(ABI, *Binary, UseSoftRegisterStateDeductions);
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

  revng_log(Log, "Conversion successful:\n" << serializeToString(NewType));

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
  // Rely onto the register state deduction to make sure no "holes" are
  // present in-between the argument registers.
  abi::RegisterState::Map Map(model::ABI::getArchitecture(ABI.ABI()));
  for (const model::NamedTypedRegister &Reg : Registers)
    Map[Reg.Location()].IsUsedForPassingArguments = abi::RegisterState::Yes;
  abi::RegisterState::Map DeductionResults = Map;
  if (UseSoftRegisterStateDeductions) {
    std::optional MaybeDeductionResults = ABI.tryDeducingRegisterState(Map);
    if (MaybeDeductionResults.has_value())
      DeductionResults = std::move(MaybeDeductionResults.value());
    else
      return std::nullopt;
  } else {
    DeductionResults = ABI.enforceRegisterState(Map);
  }

  llvm::SmallVector<model::Register::Values, 8> ArgumentRegisters;
  for (auto [Register, Pair] : DeductionResults)
    if (abi::RegisterState::shouldEmit(Pair.IsUsedForPassingArguments))
      ArgumentRegisters.emplace_back(Register);

  // But just knowing which registers we need is not sufficient, we also have to
  // order them properly.
  auto Ordered = ABI.sortArguments(ArgumentRegisters);

  llvm::SmallVector<model::Argument, 8> Result;
  for (model::Register::Values Register : Ordered) {
    auto CurrentRegisterIterator = Registers.find(Register);
    if (CurrentRegisterIterator != Registers.end()) {
      // If the current register is confirmed to be in use, convert it into
      // an argument while preserving its type and metadata.
      model::Argument &Current = Result.emplace_back();
      Current.CustomName() = CurrentRegisterIterator->CustomName();

      if (CurrentRegisterIterator->Type().UnqualifiedType().get() != nullptr)
        Current.Type() = CurrentRegisterIterator->Type();
      else
        Current.Type() = { Bucket.defaultRegisterType(Register), {} };
    } else {
      // This register is unused but we still have to create an argument
      // for it. Otherwise the resulting function will be semantically different
      // from the input one.
      //
      // Also, if the indicator for this "hole" is not preserved, there will be
      // no way to recreate it at any point in the future, when it's being
      // converted back to the representation similar to original (e.g. when
      // producing the `Layout` for this function).
      Result.emplace_back().Type() = { Bucket.genericRegisterType(Register),
                                       {} };
    }
  }

  // Rewrite all the indices to make sure they are consistent.
  for (auto Pair : llvm::enumerate(Result))
    Pair.value().Index() = Pair.index();

  return Result;
}

std::optional<llvm::SmallVector<model::Argument, 8>>
TCC::tryConvertingStackArguments(model::QualifiedType StackArgumentTypes,
                                 size_t IndexOffset) {
  if (!StackArgumentTypes.UnqualifiedType().isValid()) {
    // If there is no type, it means that the importer responsible for this
    // function didn't detect any stack arguments and avoided creating
    // a new empty type.
    return llvm::SmallVector<model::Argument, 8>{};
  }

  // Qualifiers here are not allowed since this must point to a simple struct.
  revng_assert(StackArgumentTypes.Qualifiers().empty());
  auto *Unqualified = StackArgumentTypes.UnqualifiedType().get();

  // If the struct is empty, it indicates that there are no stack arguments.
  const model::StructType &Types = *llvm::cast<model::StructType>(Unqualified);
  if (Types.Fields().empty()) {
    revng_assert(Types.size() == 0);
    return llvm::SmallVector<model::Argument, 8>{};
  }

  LoggerIndent Indentation(Log);

  // Compute the alignment of the first argument.
  auto CurrentAlignment = *ABI.alignment(Types.Fields().begin()->Type());
  if (!llvm::isPowerOf2_64(CurrentAlignment)) {
    revng_log(Log,
              "The natural alignment of a type is not a power of two:\n"
                << serializeToString(Types.Fields().begin()->Type()));
    return std::nullopt;
  }

  // Process each of the fields (except for the very last one),
  // converting them into arguments separately.
  llvm::SmallVector<model::Argument, 8> Result;
  for (auto Iterator = Types.Fields().begin();
       Iterator != std::prev(Types.Fields().end());
       ++Iterator) {
    const model::StructField &CurrentArgument = *Iterator;
    std::optional<std::uint64_t> MaybeSize = CurrentArgument.Type().size();
    revng_assert(MaybeSize.has_value() && MaybeSize.value() != 0);

    const std::uint64_t RegisterSize = model::ABI::getPointerSize(ABI.ABI());
    uint64_t CurrentSize = paddedSizeOnStack(MaybeSize.value(), RegisterSize);

    const model::StructField &NextArgument = *std::next(Iterator);
    std::uint64_t NextAlignment = *ABI.alignment(NextArgument.Type());
    if (!llvm::isPowerOf2_64(NextAlignment)) {
      revng_log(Log,
                "The natural alignment of a type is not a power of two:\n"
                  << serializeToString(NextArgument.Type()));
      return std::nullopt;
    }

    OverflowSafeInt Offset = CurrentArgument.Offset();
    Offset += CurrentSize;
    if (!Offset) {
      // Abandon if offset overflows.
      revng_log(Log, "Integer overflow when calculating field offsets.");
      return std::nullopt;
    }

    if (*Offset != NextArgument.Offset() || *Offset % NextAlignment != 0) {
      revng_log(Log,
                "The natural alignment of a type would make it impossible "
                "to represent as CABI, abandon the conversion.\nTODO: we "
                "might want to preprocess such functions and manually "
                "\"fill\" the holes in before attempting the conversion.");
      return std::nullopt;
    }

    // Create the argument from this field.
    model::Argument &New = Result.emplace_back();
    model::copyMetadata(New, CurrentArgument);
    New.Index() = IndexOffset++;

    // TODO: ensure that the type is in fact naturally aligned
    New.Type() = CurrentArgument.Type();

    CurrentAlignment = NextAlignment;
  }

  // And don't forget the very last field.
  const model::StructField &LastArgument = *std::prev(Types.Fields().end());
  model::Argument &New = Result.emplace_back();
  model::copyMetadata(New, LastArgument);
  New.Index() = IndexOffset++;

  // TODO: ensure that the type is in fact naturally aligned
  New.Type() = LastArgument.Type();

  // Make sure the size does not contradict the final alignment.
  std::uint64_t FullAlignment = *ABI.alignment(StackArgumentTypes);
  if (!llvm::isPowerOf2_64(FullAlignment)) {
    revng_log(Log,
              "The natural alignment of a type is not a power of two:\n"
                << serializeToString(Types));
    return std::nullopt;
  }

  return Result;
}

std::optional<model::QualifiedType>
TCC::tryConvertingReturnValue(const ReturnValueRegisters &Registers) {
  if (Registers.size() == 0) {
    // The function doesn't return anything.
    return model::QualifiedType{
      Bucket.getPrimitiveType(model::PrimitiveTypeKind::Void, 0), {}
    };
  }

  // We only convert register-based return values: those that are returned using
  // a pointer to memory readied by the callee are technically fine without any
  // intervention (they are just `void` functions that modify some object passed
  // into them with a pointer.
  //
  // We might want to handle this in a different way under some architectures
  // (i.e. ARM64 because it uses a separate `PointerToCopyLocation` register),
  // but for now the dumb approach should suffice.

  abi::RegisterState::Map Map(model::ABI::getArchitecture(ABI.ABI()));
  for (const model::TypedRegister &Register : Registers)
    Map[Register.Location()].IsUsedForReturningValues = abi::RegisterState::Yes;
  abi::RegisterState::Map DeductionResults = Map;
  if (UseSoftRegisterStateDeductions) {
    std::optional MaybeDeductionResults = ABI.tryDeducingRegisterState(Map);
    if (MaybeDeductionResults.has_value())
      DeductionResults = std::move(MaybeDeductionResults.value());
    else
      return std::nullopt;
  } else {
    DeductionResults = ABI.enforceRegisterState(Map);
  }

  llvm::SmallSet<model::Register::Values, 8> ReturnValueRegisters;
  for (auto [Register, Pair] : DeductionResults)
    if (abi::RegisterState::shouldEmit(Pair.IsUsedForReturningValues))
      ReturnValueRegisters.insert(Register);
  auto Ordered = ABI.sortReturnValues(ReturnValueRegisters);

  if (Ordered.size() == 1) {
    if (auto Iter = Registers.find(*Ordered.begin()); Iter != Registers.end()) {
      // Only one register is used, just return its type.
      return Iter->Type();
    } else {
      // One register is used but its type cannot be obtained.
      // Create a register sized return type instead.
      return model::QualifiedType(Bucket.defaultRegisterType(*Ordered.begin()),
                                  {});
    }
  } else {
    // Multiple registers mean that it should probably be a struct.
    //
    // \note: it's also possible that it's just a big primitive,
    // \todo: look into supporting those.
    auto [ReturnType, ReturnTypePath] = Bucket.makeType<model::StructType>();
    for (auto Register : Ordered) {
      // Make a separate field for each register.
      model::StructField Field;
      Field.Offset() = ReturnType.Size();
      if (auto Iter = Registers.find(*Ordered.begin()); Iter != Registers.end())
        Field.Type() = Iter->Type();
      else
        Field.Type() = { Bucket.genericRegisterType(*Ordered.begin()), {} };

      std::optional<std::uint64_t> FieldSize = Field.Type().size();
      revng_assert(FieldSize.has_value() && FieldSize.value() != 0);

      // Round the next offset based on the natural alignment.
      std::optional<std::uint64_t> Alignment = ABI.alignment(Field.Type());
      revng_assert(Alignment.has_value() && Alignment.value() != 0);
      ReturnType.Size() += (*Alignment - ReturnType.Size() % *Alignment);

      // Insert the field
      ReturnType.Fields().insert(std::move(Field));

      // Update the total struct size: insert some padding if necessary.
      std::uint64_t RegisterSize = model::ABI::getPointerSize(ABI.ABI());
      ReturnType.Size() += paddedSizeOnStack(FieldSize.value(), RegisterSize);
    }

    revng_assert(ReturnType.Size() != 0 && !ReturnType.Fields().empty());
    return model::QualifiedType(ReturnTypePath, {});
  }
}

} // namespace abi::FunctionType
