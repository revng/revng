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
#include "revng/Model/QualifiedType.h"
#include "revng/Support/Debug.h"

#include "ValueDistributor.h"

static Logger Log("function-type-conversion-to-raw");

namespace abi::FunctionType {

class ToRawConverter {
private:
  using CFT = const model::CABIFunctionType &;
  using CFTArguments = decltype(std::declval<CFT>().Arguments());

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
  DistributedValue
  distributeReturnValue(const model::QualifiedType &ReturnValueType) const {
    return ReturnValueDistributor(ABI).returnValue(ReturnValueType);
  }

  /// Helper used for deciding how an arbitrary set of arguments should be
  /// distributed across registers and the stack accordingly to the \ref ABI.
  ///
  /// \param Arguments The list of arguments to distribute.
  /// \param HasReturnValueLocation `true` if the first argument slot should
  ///        be occupied by a shadow return value, `false` otherwise.
  ///
  /// \return Information about registers and stack that are to be used to
  ///         pass said arguments.
  DistributedValues distributeArguments(CFTArguments Arguments,
                                        bool HasReturnValueLocation) const;

public:
  uint64_t finalStackOffset(uint64_t SizeOfArgumentsOnStack) const;
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
  std::optional<uint64_t> MaybeSize = ArgumentType.size();
  uint64_t TargetSize = model::Register::getSize(Register);

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

  // Since this conversion cannot fail, nothing prevents us from creating
  // the result type right away.
  auto [NewType, NewTypePath] = Binary->makeType<model::RawFunctionType>();
  model::copyMetadata(NewType, FunctionType);

  model::StructType StackArguments;
  uint64_t CurrentStackOffset = 0;

  // Since shadow arguments are a concern, we need to deal with the return
  // value first.
  auto ReturnValue = distributeReturnValue(FunctionType.ReturnType());
  if (!ReturnValue.Registers.empty()) {
    revng_assert(ReturnValue.SizeOnStack == 0);

    // The return value uses registers: pass them through to the new type.
    for (model::Register::Values Register : ReturnValue.Registers) {
      model::NamedTypedRegister Converted;
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

    const model::Architecture::Values Architecture = Binary->Architecture();
    auto ReturnType = FunctionType.ReturnType().getPointerTo(Architecture);

    if (ABI.ReturnValueLocationRegister() != model::Register::Invalid) {
      model::NamedTypedRegister InputPointer(ABI.ReturnValueLocationRegister());
      InputPointer.Type() = ReturnType;

      revng_log(Log,
                "Adding a register argument to represent the return value "
                "location:\n"
                  << serializeToString(InputPointer));
      NewType.Arguments().emplace(std::move(InputPointer));
    } else if (ABI.ReturnValueLocationOnStack()) {
      model::StructField InputPointer(0);
      InputPointer.Type() = ReturnType;

      revng_log(Log,
                "Adding a stack argument to represent the return value "
                "location:\n"
                  << serializeToString(InputPointer));
      StackArguments.Fields().emplace(std::move(InputPointer));
      CurrentStackOffset += ABI.getPointerSize();
    } else {
      revng_abort("Current ABI doesn't support big return values.");
    }

    revng_log(Log,
              "Return value is returned through a shadow-argument-pointer:\n"
                << serializeToString(ReturnType));

    revng_assert(!ABI.GeneralPurposeReturnValueRegisters().empty());
    auto FirstRegister = ABI.GeneralPurposeReturnValueRegisters()[0];
    model::NamedTypedRegister OutputPointer(FirstRegister);
    OutputPointer.Type() = std::move(ReturnType);
    NewType.ReturnValues().emplace(std::move(OutputPointer));
  } else {
    // The function returns `void`: no need to do anything special.
    revng_log(Log, "Return value is `void`\n");
  }

  // Now that return value is figured out, the arguments are next.
  auto Arguments = distributeArguments(FunctionType.Arguments(),
                                       ReturnValue.SizeOnStack != 0);
  uint64_t Index = 0;
  for (const DistributedValue &Distributed : Arguments) {
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
        uint64_t InternalStackOffset = CurrentStackOffset;

        uint64_t OccupiedSpace = *Argument.Type().size();
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
          uint64_t PointerSize = ABI.getPointerSize();
          auto T = Binary->getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                            PointerSize);
          uint64_t RC = Distributed.Registers.size();
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

    // If the resulting type is a single stack-like struct, reuse it.
    if (StackArguments.Fields().size() == 1) {
      auto FirstFieldType = StackArguments.Fields().begin()->Type();
      if (FirstFieldType.Qualifiers().empty()) {
        auto *Unqualified = FirstFieldType.UnqualifiedType().get();
        if (auto *Struct = llvm::dyn_cast<model::StructType>(Unqualified))
          if (Struct->Size() == StackArguments.Size())
            NewType.StackArgumentsType() = Binary->getTypePath(Struct->key());
      }
    }

    if (NewType.StackArgumentsType().empty()) {
      auto [_, Path] = Binary->makeType<model::StructType>(StackArguments);
      NewType.StackArgumentsType() = Path;
    }
  }

  // Set the final stack offset
  NewType.FinalStackOffset() = finalStackOffset(CurrentStackOffset);

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

uint64_t
ToRawConverter::finalStackOffset(uint64_t SizeOfArgumentsOnStack) const {
  const auto Architecture = model::ABI::getArchitecture(ABI.ABI());
  uint64_t Result = model::Architecture::getCallPushSize(Architecture);

  if (ABI.CalleeIsResponsibleForStackCleanup()) {
    Result += SizeOfArgumentsOnStack;

    // TODO: take return values into the account.

    // TODO: take shadow space into the account if relevant.

    revng_assert(llvm::isPowerOf2_64(ABI.StackAlignment()));
    Result += ABI.StackAlignment() - 1;
    Result &= ~(ABI.StackAlignment() - 1);
  }

  return Result;
}

DistributedValues
ToRawConverter::distributeArguments(CFTArguments Arguments,
                                    bool HasReturnValueLocationArgument) const {
  uint64_t SkippedRegisterCount = 0;
  if (HasReturnValueLocationArgument == true)
    if (const auto &GPRs = ABI.GeneralPurposeArgumentRegisters(); !GPRs.empty())
      if (ABI.ReturnValueLocationRegister() == GPRs[0])
        SkippedRegisterCount = 1;

  ArgumentDistributor Distributor(ABI);
  Distributor.UsedGeneralPurposeRegisterCount = SkippedRegisterCount;
  Distributor.ArgumentIndex = SkippedRegisterCount;

  DistributedValues Result;
  for (const model::Argument &Argument : Arguments)
    std::ranges::move(Distributor.nextArgument(Argument.Type()),
                      std::back_inserter(Result));
  return Result;
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
  uint64_t CurrentStackOffset = 0;
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
  uint64_t Index = 0;
  for (const DistributedValue &Distributed : Converted) {
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

  FinalStackOffset = Converter.finalStackOffset(CurrentStackOffset);
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
  for (const model::NamedTypedRegister &Register : Function.ReturnValues()) {
    auto &ReturnValue = ReturnValues.emplace_back();
    ReturnValue.Registers = { Register.Location() };
    ReturnValue.Type = Register.Type();
  }

  // Lay stack arguments out.
  if (not Function.StackArgumentsType().empty()) {
    const model::TypePath &StackArgTypeRef = Function.StackArgumentsType();
    auto *StackStruct = llvm::cast<model::StructType>(StackArgTypeRef.get());

    auto &Argument = Arguments.emplace_back();

    // Stack argument is always passed by pointer for RawFunctionType
    Argument.Type = model::QualifiedType{ StackArgTypeRef, {} };
    Argument.Kind = ArgumentKind::ReferenceToAggregate;

    // Record the size
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
