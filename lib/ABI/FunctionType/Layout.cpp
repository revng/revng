/// \file Layout.cpp

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

using RegisterVector = llvm::SmallVector<model::Register::Values, 2>;

/// The internal representation of the argument shared between both
/// the to-raw conversion and the layout.
struct DistributedArgument {
  /// A list of registers the argument uses.
  RegisterVector Registers = {};

  /// The total size of the argument (including padding if necessary) in bytes.
  std::uint64_t Size = 0;

  /// The size of the piece of the argument placed on the stack.
  /// \note: has to be equal to `0` or `this->Size` for any ABI for which
  ///        `abi::Definition::ArgumentsCanBeSplitBetweenRegistersAndStack()`
  ///        returns `false`. Otherwise, it has to be an integer value, such
  ///        that `(0 <= SizeOnStack <= this->Size)` is true.
  std::uint64_t SizeOnStack = 0;

  /// Mark this argument as a padding argument, which means an unused location
  /// (either a register or a piece of the stack) which needs to be seen as
  /// a separate argument to be able to place all the following arguments
  /// in the correct positions.
  ///
  /// The "padding" arguments are emitted as normal arguments in RawFunctionType
  /// but are omitted in `Layout`.
  bool RepresentsPadding = false;
};
using DistributedArguments = llvm::SmallVector<DistributedArgument, 8>;

using RegisterSpan = std::span<const model::Register::Values>;
using ArgumentSet = TrackingSortedVector<model::Argument>;

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

  /// Helper for converting a single argument into a "distributed" state.
  ///
  /// \param Type The type of the arguments.
  /// \param Registers The list of registers allowed for usage for the given
  ///        Argument type.
  /// \param OccupiedRegisterCount The count of registers in \ref Registers
  ///        Container that are already occupied.
  /// \param AllowedRegisterLimit The maximum number of registers available to
  ///        use for the current argument.
  /// \param AllowPuttingPartOfAnArgumentOnStack Specifies whether the ABI
  ///        allows splitting the argument placing parts of it both into
  ///        the registers and onto the stack.
  ///
  /// \return New distributed argument list (could be multiple if padding
  ///         arguments were required) and the new value of
  ///         \ref OccupiedRegisterCount after this argument is distributed.
  std::pair<DistributedArguments, std::size_t>
  distributeOne(model::QualifiedType Type,
                RegisterSpan Registers,
                std::size_t OccupiedRegisterCount,
                std::size_t AllowedRegisterLimit,
                bool AllowPuttingPartOfAnArgumentOnStack) const;

public:
  std::uint64_t finalStackOffset(const DistributedArguments &Arguments) const;
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

  model::StructType StackArguments;
  std::uint64_t CurrentStackOffset = 0;

  // Now that return value is figured out, the arguments are next.
  auto Arguments = distributeArguments(FunctionType.Arguments(),
                                       ReturnValue.SizeOnStack != 0);
  std::size_t Index = 0;
  for (const DistributedArgument &Distributed : Arguments) {
    if (!Distributed.RepresentsPadding) {
      // Transfer the register arguments first.
      for (model::Register::Values Register : Distributed.Registers) {
        model::NamedTypedRegister Argument(Register);

        const auto &ArgumentType = FunctionType.Arguments().at(Index).Type();
        const bool IsSingleRegister = Distributed.Registers.size() == 1
                                      && Distributed.SizeOnStack == 0;
        Argument.Type() = IsSingleRegister ?
                            chooseType(ArgumentType, Register, *Binary) :
                            genericRegisterType(Register, *Binary);

        revng_log(Log,
                  "Adding an argument register:\n"
                    << serializeToString(Register));
        NewType.Arguments().emplace(Argument);
      }

      // Then stack arguments.
      if (Distributed.SizeOnStack != 0) {
        auto ArgumentIterator = FunctionType.Arguments().find(Index);
        revng_assert(ArgumentIterator != FunctionType.Arguments().end());
        const model::Argument &Argument = *ArgumentIterator;

        // Round the next offset based on the natural alignment.
        CurrentStackOffset = ABI.alignedOffset(CurrentStackOffset,
                                               Argument.Type());
        std::uint64_t InternalStackOffset = CurrentStackOffset;

        std::uint64_t OccupiedSpace = *Argument.Type().size();
        if (Distributed.Registers.empty()) {
          // A stack-only argument: convert it into a struct field.
          model::StructField Field;
          model::copyMetadata(Field, Argument);
          Field.Offset() = CurrentStackOffset;
          Field.Type() = Argument.Type();

          revng_log(Log,
                    "Adding a stack argument:\n"
                      << serializeToString(Field));
          StackArguments.Fields().emplace(std::move(Field));
        } else {
          // A piece of the argument is in registers, the rest is on the stack.
          // TODO: there must be more efficient way to handle these, but for
          //       the time being, just replace the argument type with `Generic`
          std::size_t PointerSize = ABI.getPointerSize();
          auto T = Binary->getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                            PointerSize);
          std::size_t RC = Distributed.Registers.size();
          OccupiedSpace -= PointerSize * RC;
          std::ptrdiff_t LeftoverSpace = OccupiedSpace;
          while (LeftoverSpace > 0) {
            model::StructField Field;
            Field.Offset() = InternalStackOffset;
            Field.Type() = model::QualifiedType(T, {});

            revng_log(Log,
                      "Adding a stack argument piece:\n"
                        << serializeToString(Field));
            StackArguments.Fields().emplace(std::move(Field));

            InternalStackOffset += PointerSize;
            LeftoverSpace -= PointerSize;
          }
        }

        // Compute the full size of the argument (including padding if needed),
        // so that the next argument is not placed into this occupied space.
        CurrentStackOffset += ABI.paddedSizeOnStack(OccupiedSpace);
        revng_assert(InternalStackOffset <= CurrentStackOffset);
      }

      ++Index;
    } else {
      // This is just a padding argument.
      for (model::Register::Values Register : Distributed.Registers) {
        model::NamedTypedRegister Argument(Register);
        Argument.Type() = genericRegisterType(Register, *Binary);

        revng_log(Log,
                  "Adding a padding argument:\n"
                    << serializeToString(Argument));
        NewType.Arguments().emplace(std::move(Argument));
      }
    }
  }

  // If the stack argument struct is not empty, record it into the model.
  if (CurrentStackOffset != 0) {
    revng_assert(!StackArguments.Fields().empty());
    StackArguments.Size() = CurrentStackOffset;

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

  revng_assert(NewType.verify(true));

  revng_log(Log, "Conversion successful:\n" << serializeToString(NewType));

  // To finish up the conversion, remove all the references to the old type by
  // carefully replacing them with references to the new one.
  replaceAllUsesWith(FunctionType.key(), NewTypePath, Binary);

  // And don't forget to remove the old type.
  Binary->Types().erase(FunctionType.key());

  return NewTypePath;
}

std::uint64_t
ToRawConverter::finalStackOffset(const DistributedArguments &Arguments) const {
  const auto Architecture = model::ABI::getArchitecture(ABI.ABI());
  std::uint64_t Result = model::Architecture::getCallPushSize(Architecture);

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

std::pair<DistributedArguments, std::size_t>
ToRawConverter::distributeOne(model::QualifiedType Type,
                              RegisterSpan Registers,
                              std::size_t OccupiedRegisterCount,
                              std::size_t AllowedRegisterLimit,
                              bool AllowPuttingPartOfAnArgumentOnStack) const {
  DistributedArguments Result;

  LoggerIndent Indentation(Log);
  revng_log(Log, "Distributing a value between the registers and the stack.");
  revng_log(Log,
            OccupiedRegisterCount << " registers are already occupied, while "
                                     "only "
                                  << AllowedRegisterLimit
                                  << " registers are available to be used.");
  revng_log(Log, "The total number of registers is " << Registers.size());

  std::uint64_t Size = *Type.size();
  std::uint64_t Alignment = *ABI.alignment(Type);
  revng_log(Log, "The type:\n" << serializeToString(Type));
  revng_log(Log,
            "Its size is " << Size << " and its alignment is " << Alignment
                           << ".");

  // Precompute the last register allowed to be used.
  std::size_t LastRegister = OccupiedRegisterCount + AllowedRegisterLimit;
  if (LastRegister > Registers.size())
    LastRegister = Registers.size();

  // Define the counters: one for the number of registers the current value
  // could occupy, and one for the total size of said registers.
  std::size_t ConsideredRegisterCounter = OccupiedRegisterCount;
  std::size_t SizeCounter = 0;

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
  std::optional<std::size_t> PaddingRegisterIndex = std::nullopt;
  if (ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
    const std::uint64_t PointerSize = ABI.getPointerSize();
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
      DistributedArgument &Padding = Result.emplace_back();
      Padding.Registers = { Registers[OccupiedRegisterCount++] };
      Padding.Size = model::Register::getSize(Padding.Registers[0]);
      Padding.RepresentsPadding = true;

      revng_assert(SizeCounter >= Padding.Size);
      SizeCounter -= Padding.Size;
      if (ConsideredRegisterCounter < LastRegister)
        ++ConsideredRegisterCounter;
    }
  }

  DistributedArgument &DA = Result.emplace_back();
  DA.Size = Size;

  if (SizeCounter >= Size) {
    // This a register-only argument, add the registers.
    for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
      DA.Registers.emplace_back(Registers[I]);
    DA.SizeOnStack = 0;
  } else if (AllowPuttingPartOfAnArgumentOnStack
             && ConsideredRegisterCounter == LastRegister) {
    // This argument is split among the registers and the stack.
    for (size_t I = OccupiedRegisterCount; I < ConsideredRegisterCounter; ++I)
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

using ASet = ArgumentSet;
DistributedArguments
ToRawConverter::distributeNonPositionBasedArguments(const ASet &Arguments,
                                                    size_t SkippedCount) const {
  DistributedArguments Result;

  std::size_t UsedVectorRegisterCounter = 0;
  std::size_t UsedGeneralPurposeRegisterCounter = SkippedCount;
  for (const model::Argument &Argument : Arguments) {
    bool CanSplit = ABI.ArgumentsCanBeSplitBetweenRegistersAndStack();

    std::size_t RegisterLimit = 0;
    std::size_t *RegisterCounter = nullptr;
    std::span<const model::Register::Values> RegisterList;
    if (Argument.Type().isFloat()) {
      RegisterList = ABI.VectorArgumentRegisters();
      RegisterCounter = &UsedVectorRegisterCounter;

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
        DistributedArgument &VectorArgument = Result.emplace_back();
        model::Register::Values Register = RegisterList[(*RegisterCounter)++];
        VectorArgument.Registers.emplace_back(Register);
        continue;
      } else {
        // If there are no free registers left, explicitly set the limit to 0,
        // so that the default argument distribution routine puts it on
        // the stack.
        RegisterLimit = 0;
      }

      // Explicitly disallow splitting vector arguments across the registers
      // and the stack.
      CanSplit = false;
    } else {
      RegisterList = ABI.GeneralPurposeArgumentRegisters();
      RegisterCounter = &UsedGeneralPurposeRegisterCounter;
      RegisterLimit = Argument.Type().isScalar() ?
                        ABI.MaximumGPRsPerScalarArgument() :
                        ABI.MaximumGPRsPerAggregateArgument();
    }

    auto [Arguments, NextRegisterIndex] = distributeOne(Argument.Type(),
                                                        RegisterList,
                                                        *RegisterCounter,
                                                        RegisterLimit,
                                                        CanSplit);

    // Verify that the next register makes sense.
    auto VerifyNextRegisterIndex = [&](std::size_t Current, std::size_t Next) {
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

    // All good - insert the arguments.
    for (DistributedArgument &&Argument : as_rvalue(Arguments))
      Result.emplace_back(std::move(Argument));
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

using model::QualifiedType;
DistributedArgument
ToRawConverter::distributeReturnValue(const QualifiedType &ReturnValue) const {
  if (ReturnValue.isVoid())
    return DistributedArgument{};

  std::size_t Limit = 0;
  std::span<const model::Register::Values> RegisterList;
  if (ReturnValue.isFloat()) {
    RegisterList = ABI.VectorReturnValueRegisters();

    // For now replace unsupported floating point return values with `void`
    // The main offenders are the values returned in `st0`.
    // TODO: handle this properly.
    if (RegisterList.empty())
      return DistributedArgument{};

    // TODO: replace this the explicit single register limit with an abi-defined
    // value. For more information see the relevant comment in
    // `distributeRegisterArguments`.
    Limit = 1;
  } else {
    RegisterList = ABI.GeneralPurposeReturnValueRegisters();
    Limit = ReturnValue.isScalar() ? ABI.MaximumGPRsPerScalarReturnValue() :
                                     ABI.MaximumGPRsPerAggregateReturnValue();
  }

  auto [Result, _] = distributeOne(ReturnValue, RegisterList, 0, Limit, false);
  revng_assert(Result.size() == 1);
  return Result[0];
}

model::TypePath convertToRaw(const model::CABIFunctionType &FunctionType,
                             TupleTree<model::Binary> &Binary) {
  ToRawConverter ToRaw(abi::Definition::get(FunctionType.ABI()));
  return ToRaw.convert(FunctionType, Binary);
}

Layout::Layout(const model::CABIFunctionType &Function) {
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  ToRawConverter Converter(ABI);

  //
  // Handle return values first (since it might mean adding an extra argument).
  //

  bool HasShadowPointerToAggregateReturnValue = false;
  std::uint64_t CurrentStackOffset = 0;
  const auto Architecture = model::ABI::getArchitecture(Function.ABI());
  auto RV = Converter.distributeReturnValue(Function.ReturnType());
  if (RV.SizeOnStack == 0) {
    if (not Function.ReturnType().isVoid()) {
      // Nothing on the stack, the return value fits into the registers.
      auto &ReturnValue = ReturnValues.emplace_back();
      ReturnValue.Type = Function.ReturnType();
      ReturnValue.Registers = std::move(RV.Registers);
    }
  } else {
    revng_assert(RV.Registers.empty(),
                 "Register and stack return values should never be present "
                 "at the same time.");

    // Add an argument to represent the pointer to the return value location.
    Argument &RVLocationIn = Arguments.emplace_back();
    RVLocationIn.Type = Function.ReturnType().getPointerTo(Architecture);
    RVLocationIn.Kind = ArgumentKind::ShadowPointerToAggregateReturnValue;
    HasShadowPointerToAggregateReturnValue = true;

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

  //
  // Then distribute the arguments.
  //

  auto Converted = Converter.distributeArguments(Function.Arguments(),
                                                 RV.SizeOnStack != 0);
  revng_assert(Converted.size() >= Function.Arguments().size());
  std::size_t Index = 0;
  for (const DistributedArgument &Distributed : Converted) {
    if (!Distributed.RepresentsPadding) {
      Argument &Current = Arguments.emplace_back();
      const auto &ArgumentType = Function.Arguments().at(Index).Type();

      // Disambiguate scalar and aggregate arguments.
      // Scalars are passed by value, aggregates - by pointer.
      Current.Type = ArgumentType;
      if (ArgumentType.isScalar())
        Current.Kind = ArgumentKind::Scalar;
      else
        Current.Kind = ArgumentKind::ReferenceToAggregate;

      Current.Registers = std::move(Distributed.Registers);
      if (Distributed.SizeOnStack != 0) {
        // The argument has a part (or is placed entirely) on the stack.
        Current.Stack = Layout::Argument::StackSpan{};

        // Round the offset based on the natural alignment,
        CurrentStackOffset = ABI.alignedOffset(CurrentStackOffset,
                                               ArgumentType);
        Current.Stack->Offset = CurrentStackOffset;

        // And carry the size over unchanged.
        Current.Stack->Size = Distributed.SizeOnStack;
        CurrentStackOffset += Current.Stack->Size;
      }

      ++Index;
    }
  }

  if (HasShadowPointerToAggregateReturnValue)
    revng_assert(Arguments.size() == Function.Arguments().size() + 1);
  else
    revng_assert(Arguments.size() == Function.Arguments().size());

  CalleeSavedRegisters.resize(ABI.CalleeSavedRegisters().size());
  llvm::copy(ABI.CalleeSavedRegisters(), CalleeSavedRegisters.begin());

  FinalStackOffset = Converter.finalStackOffset(Converted);
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

  // If we have more than one return value, each return value should take at
  // most a single register
  if (ReturnValues.size() > 1)
    for (const ReturnValue &ReturnValue : ReturnValues)
      if (ReturnValue.Registers.size() > 1)
        return false;

  return true;
}

std::size_t Layout::argumentRegisterCount() const {
  std::size_t Result = 0;

  for (const auto &Argument : Arguments)
    Result += Argument.Registers.size();

  return Result;
}

std::size_t Layout::returnValueRegisterCount() const {
  std::size_t Result = 0;

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
