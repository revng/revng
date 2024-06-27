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
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/NamedEnumScalarTraits.h"

#include "ValueDistributor.h"

static Logger Log("function-type-conversion-to-raw");

namespace abi::FunctionType {

class ToRawConverter {
private:
  using CFT = const model::CABIFunctionDefinition &;
  using CFTArguments = decltype(std::declval<CFT>().Arguments());

private:
  const abi::Definition &ABI;

public:
  explicit ToRawConverter(const abi::Definition &ABI) : ABI(ABI) {
    revng_assert(ABI.verify());
  }

public:
  /// Entry point for the `toRaw` conversion.
  model::UpcastableType
  convert(const model::CABIFunctionDefinition &FunctionType,
          TupleTree<model::Binary> &Binary) const;

  /// Helper used for deciding how an arbitrary return type should be
  /// distributed across registers and the stack accordingly to the \ref ABI.
  ///
  /// \param ReturnValueType The model type that should be returned.
  /// \return Information about registers and stack that are to be used to
  ///         return the said type.
  DistributedValue
  distributeReturnValue(const model::Type &ReturnValueType) const {
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

  uint64_t
  combinedStackArgumentSize(const model::CABIFunctionDefinition &) const;
};

model::UpcastableType
ToRawConverter::convert(const model::CABIFunctionDefinition &FunctionType,
                        TupleTree<model::Binary> &Binary) const {
  revng_log(Log,
            "Converting a `CABIFunctionDefinition` to "
            "`RawFunctionDefinition`.");
  revng_log(Log, "Original type:\n" << serializeToString(FunctionType));
  LoggerIndent Indentation(Log);

  // Since this conversion cannot fail, nothing prevents us from creating
  // the result type right away.
  auto [NewPrototype, NewType] = Binary->makeRawFunctionDefinition();

  revng_assert(FunctionType.ABI() != model::ABI::Invalid);
  NewPrototype.Architecture() = model::ABI::getArchitecture(FunctionType.ABI());
  model::copyMetadata(NewPrototype, FunctionType);

  model::StructDefinition StackArguments;
  uint64_t StackStructSize = 0;

  // Since shadow arguments are a concern, we need to deal with the return
  // value first.
  bool UsesSPTAR = false;
  if (!FunctionType.ReturnType().isEmpty()) {
    auto ReturnValue = distributeReturnValue(*FunctionType.ReturnType());
    if (!ReturnValue.Registers.empty()) {
      revng_assert(ReturnValue.SizeOnStack == 0);

      // The return value uses registers: pass them through to the new type.
      for (model::Register::Values Register : ReturnValue.Registers) {
        model::NamedTypedRegister Converted;
        Converted.Location() = Register;

        const model::UpcastableType &ReturnType = FunctionType.ReturnType();
        if (ReturnValue.Registers.size() > 1) {
          Converted.Type() = model::PrimitiveType::makeGeneric(Register);
        } else if (ReturnValue.UsesPointerToCopy == true) {
          revng_assert(*ReturnType->size() > ABI.getPointerSize());
          Converted.Type() = model::PointerType::make(ReturnType.copy(),
                                                      ABI.getArchitecture());
        } else if (ReturnType->isScalar()) {
          Converted.Type() = ReturnType.copy();
        } else {
          Converted.Type() = model::PrimitiveType::make(Register);
        }

        revng_log(Log,
                  "Adding a return value register:\n"
                    << serializeToString(Register));

        NewPrototype.ReturnValues().emplace(Converted);
      }
    } else if (ReturnValue.Size != 0) {
      // The return value uses a pointer-to-a-location: add it as an argument.
      UsesSPTAR = true;

      auto MaybeReturnValueSize = FunctionType.ReturnType()->size();
      revng_assert(MaybeReturnValueSize != std::nullopt);
      revng_assert(ReturnValue.Size == *MaybeReturnValueSize);

      auto ReturnType = model::PointerType::make(FunctionType.ReturnType()
                                                   .copy(),
                                                 Binary->Architecture());
      if (ABI.ReturnValueLocationRegister() != model::Register::Invalid) {
        model::NamedTypedRegister
          InputPointer(ABI.ReturnValueLocationRegister());
        InputPointer.Type() = ReturnType;

        revng_log(Log,
                  "Adding a register argument to represent the return value "
                  "location:\n"
                    << serializeToString(InputPointer));
        NewPrototype.Arguments().emplace(std::move(InputPointer));
      } else if (ABI.ReturnValueLocationOnStack()) {
        model::StructField InputPointer(0);
        InputPointer.Type() = ReturnType;

        revng_log(Log,
                  "Adding a stack argument to represent the return value "
                  "location:\n"
                    << serializeToString(InputPointer));
        StackArguments.Fields().emplace(std::move(InputPointer));
        StackStructSize = ABI.getPointerSize();
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
      NewPrototype.ReturnValues().emplace(std::move(OutputPointer));
    } else {
      // The function returns `void`: no need to do anything special.
      revng_log(Log, "Return value is `void`\n");
    }
  }

  // Now that return value is figured out, the arguments are next.
  auto Arguments = distributeArguments(FunctionType.Arguments(), UsesSPTAR);
  uint64_t Index = 0;
  for (const DistributedValue &Distributed : Arguments) {
    if (!Distributed.RepresentsPadding) {
      // Transfer the register arguments first.
      for (model::Register::Values Register : Distributed.Registers) {
        model::NamedTypedRegister Argument(Register);

        const auto &ArgumentType = FunctionType.Arguments().at(Index).Type();
        uint64_t ArgumentSize = *ArgumentType->size();
        if (Distributed.Registers.size() > 1 || Distributed.SizeOnStack != 0) {
          Argument.Type() = model::PrimitiveType::makeGeneric(Register);
        } else if (Distributed.UsesPointerToCopy == true) {
          revng_assert(ArgumentSize > ABI.getPointerSize());
          Argument.Type() = model::PointerType::make(ArgumentType.copy(),
                                                     ABI.getArchitecture());
        } else if (ArgumentType->isScalar()) {
          Argument.Type() = ArgumentType;
        } else {
          Argument.Type() = model::PrimitiveType::make(Register);
        }

        revng_log(Log,
                  "Adding an argument register:\n"
                    << serializeToString(Register));
        NewPrototype.Arguments().emplace(Argument);
      }

      // Then stack arguments.
      if (Distributed.SizeOnStack != 0) {
        auto ArgumentIterator = FunctionType.Arguments().find(Index);
        revng_assert(ArgumentIterator != FunctionType.Arguments().end());
        const model::Argument &Argument = *ArgumentIterator;

        if (Distributed.Registers.empty()) {
          // A stack-only argument: convert it into a struct field.
          model::StructField Field;
          model::copyMetadata(Field, Argument);
          Field.Offset() = Distributed.OffsetOnStack;
          Field.Type() = Argument.Type().copy();
          if (Distributed.UsesPointerToCopy)
            Field.Type() = model::PointerType::make(std::move(Field.Type()),
                                                    ABI.getArchitecture());

          revng_log(Log,
                    "Adding a stack argument:\n"
                      << serializeToString(Field));
          StackArguments.Fields().emplace(std::move(Field));
        } else {
          // A piece of the argument is in registers, the rest is on the stack.
          // TODO: there must be more efficient way to handle these, but for
          //       the time being, just replace the argument type with
          //       a bunch of`Generic` words so that at least size adds up.
          uint64_t PointerSize = ABI.getPointerSize();
          revng_assert(Distributed.SizeOnStack % PointerSize == 0);
          revng_log(Log,
                    "Adding " << (Distributed.SizeOnStack / PointerSize)
                              << " stack argument pieces.\n");

          auto Type = model::PrimitiveType::make(model::PrimitiveKind::Generic,
                                                 PointerSize);
          for (uint64_t CurrentWordOffset = 0;
               CurrentWordOffset < Distributed.SizeOnStack;
               CurrentWordOffset += PointerSize) {
            model::StructField Field;
            Field.Offset() = Distributed.OffsetOnStack + CurrentWordOffset;
            Field.Type() = Type.copy();
            StackArguments.Fields().emplace(std::move(Field));
          }
        }

        StackStructSize = Distributed.OffsetOnStack + Distributed.SizeOnStack;
      }

      ++Index;
    } else {
      // This is just a padding argument.
      revng_assert(Distributed.SizeOnStack == 0,
                   "Only padding register arguments are supported here. "
                   "Stack padding is represented as normal struct padding.");
      for (model::Register::Values Register : Distributed.Registers) {
        model::NamedTypedRegister Argument(Register);
        Argument.Type() = model::PrimitiveType::makeGeneric(Register);

        revng_log(Log,
                  "Adding a padding argument:\n"
                    << serializeToString(Argument));
        NewPrototype.Arguments().emplace(std::move(Argument));
      }
    }
  }

  // If the stack argument struct is not empty, record it into the model.
  if (StackStructSize != 0) {
    revng_assert(!StackArguments.Fields().empty());
    StackArguments.Size() = StackStructSize;

    // If the resulting type is a single stack-like struct, reuse it.
    if (StackArguments.Fields().size() == 1) {
      const auto FirstFieldType = StackArguments.Fields().begin()->Type();
      if (!FirstFieldType->isTypedef()) {
        const model::StructDefinition *Struct = FirstFieldType->getStruct();
        if (Struct && Struct->Size() == StackArguments.Size())
          NewPrototype.StackArgumentsType() = std::move(FirstFieldType);
      }
    }

    // Otherwise, add a new struct to the model.
    if (NewPrototype.StackArgumentsType().isEmpty()) {
      auto [_, Path] = Binary->makeStructDefinition(std::move(StackArguments));
      NewPrototype.StackArgumentsType() = Path;
    }
  }

  // Set the final stack offset
  NewPrototype.FinalStackOffset() = finalStackOffset(StackStructSize);

  // Populate the list of preserved registers
  for (auto Inserter = NewPrototype.PreservedRegisters().batch_insert();
       model::Register::Values Register : ABI.CalleeSavedRegisters())
    Inserter.insert(Register);

  revng_assert(NewPrototype.verify(true));

  revng_log(Log, "Conversion successful:\n" << serializeToString(NewPrototype));

  // To finish up the conversion, remove all the references to the old type
  // by carefully replacing them with references to the new one.
  replaceTypeDefinition(FunctionType.key(), *NewType, Binary);

  // And don't forget to remove the old type.
  Binary->TypeDefinitions().erase(FunctionType.key());

  return std::move(NewType);
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

using TRC = ToRawConverter;
uint64_t
TRC::combinedStackArgumentSize(const model::CABIFunctionDefinition &Def) const {
  DistributedValue ReturnValue;
  if (!Def.ReturnType().isEmpty())
    ReturnValue = distributeReturnValue(*Def.ReturnType());

  ArgumentDistributor Distributor(ABI);
  if (ReturnValue.SizeOnStack != 0)
    Distributor.addShadowPointerReturnValueLocationArgument();

  for (const model::Argument &Argument : Def.Arguments())
    Distributor.nextArgument(*Argument.Type());

  return Distributor.UsedStackOffset;
}

DistributedValues
ToRawConverter::distributeArguments(CFTArguments Arguments,
                                    bool HasReturnValueLocationArgument) const {
  ArgumentDistributor Distributor(ABI);
  if (HasReturnValueLocationArgument == true)
    Distributor.addShadowPointerReturnValueLocationArgument();

  DistributedValues Result;
  for (const model::Argument &Argument : Arguments)
    std::ranges::move(Distributor.nextArgument(*Argument.Type()),
                      std::back_inserter(Result));
  return Result;
}

model::UpcastableType
convertToRaw(const model::CABIFunctionDefinition &FunctionType,
             TupleTree<model::Binary> &Binary) {
  ToRawConverter ToRaw(abi::Definition::get(FunctionType.ABI()));
  return ToRaw.convert(FunctionType, Binary);
}

Layout::Layout(const model::CABIFunctionDefinition &Function) {
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  ToRawConverter Converter(ABI);

  //
  // Handle return values first (since it might mean adding an extra argument).
  //

  bool UsesSPTAR = false;
  if (!Function.ReturnType().isEmpty()) {
    const auto Architecture = model::ABI::getArchitecture(Function.ABI());
    auto RV = Converter.distributeReturnValue(*Function.ReturnType());
    if (RV.SizeOnStack == 0) {
      if (Function.ReturnType().isEmpty()) {
        // This function returns `void`: no need to do anything special.
      } else {
        // Nothing on the stack, the return value fits into the registers.
        auto &NewRV = ReturnValues.emplace_back(Function.ReturnType().copy());
        NewRV.Registers = std::move(RV.Registers);
      }
    } else {
      revng_assert(RV.Registers.empty(),
                   "Register and stack return values should never be present "
                   "at the same time.");

      // Add an argument to represent the pointer to the return value location.
      auto Pointer = model::PointerType::make(Function.ReturnType().copy(),
                                              Architecture);
      Argument &RVLocationIn = Arguments.emplace_back(std::move(Pointer));
      RVLocationIn.Kind = ArgumentKind::ShadowPointerToAggregateReturnValue;
      UsesSPTAR = true;

      if (ABI.ReturnValueLocationRegister() != model::Register::Invalid) {
        // Return value is passed using the stack (with a pointer to the
        // location in the dedicated register).
        RVLocationIn.Registers.emplace_back(ABI.ReturnValueLocationRegister());
      } else if (ABI.ReturnValueLocationOnStack()) {
        // The location, where return value should be put in, is also
        // communicated using the stack.
        RVLocationIn.Stack = { 0, model::ABI::getPointerSize(ABI.ABI()) };
      } else {
        revng_abort("Big return values are not supported by the current ABI");
      }

      // Also return the same pointer using normal means.
      //
      // NOTE: maybe some architectures do not require this.
      // TODO: investigate.
      auto &RVLocationOut = ReturnValues.emplace_back(RVLocationIn.Type.copy());

      // To simplify selecting the register for it, use the full distribution
      // routine again, but with the pointer instead of the original type.
      auto RVOut = Converter.distributeReturnValue(*RVLocationOut.Type);
      revng_assert(RVOut.Size == model::ABI::getPointerSize(ABI.ABI()));
      revng_assert(RVOut.Registers.size() == 1);
      revng_assert(RVOut.SizeOnStack == 0);
      RVLocationOut.Registers = std::move(RVOut.Registers);
    }
  }

  //
  // Then distribute the arguments.
  //

  auto Converted = Converter.distributeArguments(Function.Arguments(),
                                                 UsesSPTAR);
  revng_assert(Converted.size() >= Function.Arguments().size());
  uint64_t Index = 0;
  uint64_t StackStructSize = 0;
  for (const DistributedValue &Distributed : Converted) {
    if (!Distributed.RepresentsPadding) {
      auto &CurrentType = Function.Arguments().at(Index).Type();
      Argument &Current = Arguments.emplace_back(CurrentType.copy());

      // Disambiguate scalar and aggregate arguments.
      // Scalars are passed by value,
      // aggregates and pointer-to-copy values - as a pointer.
      if (Distributed.UsesPointerToCopy)
        Current.Kind = ArgumentKind::PointerToCopy;
      else if (Current.Type->isScalar())
        Current.Kind = ArgumentKind::Scalar;
      else
        Current.Kind = ArgumentKind::ReferenceToAggregate;

      Current.Registers = std::move(Distributed.Registers);
      if (Distributed.SizeOnStack != 0) {
        // The argument has a part (or is placed entirely) on the stack.
        Current.Stack = Layout::Argument::StackSpan{
          .Offset = Distributed.OffsetOnStack, .Size = Distributed.SizeOnStack
        };

        StackStructSize = Distributed.OffsetOnStack + Distributed.SizeOnStack;
      }

      ++Index;
    }
  }

  if (UsesSPTAR)
    revng_assert(Arguments.size() == Function.Arguments().size() + 1);
  else
    revng_assert(Arguments.size() == Function.Arguments().size());

  CalleeSavedRegisters.resize(ABI.CalleeSavedRegisters().size());
  llvm::copy(ABI.CalleeSavedRegisters(), CalleeSavedRegisters.begin());

  FinalStackOffset = Converter.finalStackOffset(StackStructSize);
}

Layout::Layout(const model::RawFunctionDefinition &Function) {
  // Lay register arguments out.
  for (const model::NamedTypedRegister &Register : Function.Arguments()) {
    revng_assert(Register.Type()->isScalar());

    auto &Argument = Arguments.emplace_back(Register.Type().copy());
    Argument.Registers = { Register.Location() };
    Argument.Kind = ArgumentKind::Scalar;
  }

  // Lay the return value out.
  for (const model::NamedTypedRegister &Register : Function.ReturnValues()) {
    auto &ReturnValue = ReturnValues.emplace_back(Register.Type().copy());
    ReturnValue.Registers = { Register.Location() };
  }

  // Lay stack arguments out.
  if (not Function.StackArgumentsType().isEmpty()) {
    auto &NewArg = Arguments.emplace_back(Function.StackArgumentsType().copy());

    // Stack argument is always passed by pointer for RawFunctionDefinition
    NewArg.Kind = ArgumentKind::ReferenceToAggregate;

    // Record the size
    const auto &StackStruct = Function.StackArgumentsType()->toStruct();
    if (StackStruct.Size() != 0)
      NewArg.Stack = { 0, StackStruct.Size() };
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

uint64_t finalStackOffset(const model::CABIFunctionDefinition &Function) {
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  ToRawConverter Helper(ABI);

  return Helper.finalStackOffset(ABI.CalleeIsResponsibleForStackCleanup() ?
                                   Helper.combinedStackArgumentSize(Function) :
                                   0);
}

UsedRegisters usedRegisters(const model::CABIFunctionDefinition &Function) {
  UsedRegisters Result;

  // Ready the return value register data.
  const abi::Definition &ABI = abi::Definition::get(Function.ABI());
  DistributedValue RV;
  if (!Function.ReturnType().isEmpty())
    RV = ToRawConverter(ABI).distributeReturnValue(*Function.ReturnType());
  std::ranges::move(RV.Registers, std::back_inserter(Result.ReturnValues));

  // Handle shadow pointer return value gracefully.
  ArgumentDistributor Distributor(ABI);
  if (RV.SizeOnStack != 0) {
    Distributor.addShadowPointerReturnValueLocationArgument();

    revng_assert(Result.ReturnValues.empty());
    const auto &GPRs = ABI.GeneralPurposeReturnValueRegisters();
    revng_assert(!GPRs.empty());
    Result.ReturnValues.emplace_back(GPRs[0]);

    if (ABI.ReturnValueLocationRegister() != model::Register::Invalid)
      Result.Arguments.emplace_back(ABI.ReturnValueLocationRegister());
  }

  if (ABI.GeneralPurposeArgumentRegisters().empty()
      && ABI.VectorArgumentRegisters().empty()) {
    // Do not even look at the arguments if an ABI explicitly states that it
    // never uses any registers in the first place.
    return Result;
  }

  // Iterate over arguments until we are sure no further argument can use
  // any registers.
  for (const model::Argument &Argument : Function.Arguments().asVector()) {
    auto Distributed = Distributor.nextArgument(*Argument.Type());
    for (DistributedValue SingleEntry : Distributed)
      if (!SingleEntry.RepresentsPadding)
        std::ranges::move(SingleEntry.Registers,
                          std::back_inserter(Result.Arguments));

    if (!Distributor.canNextArgumentUseRegisters())
      break;
  }

  return Result;
}

} // namespace abi::FunctionType

using FTL = abi::FunctionType::Layout;
namespace FTAK = abi::FunctionType::ArgumentKind;

template<>
struct llvm::yaml::ScalarEnumerationTraits<FTAK::Values>
  : public NamedEnumScalarTraits<FTAK::Values> {};

template<>
struct llvm::yaml::MappingTraits<FTL::Argument::StackSpan> {
  static void mapping(IO &IO, FTL::Argument::StackSpan &SS) {
    IO.mapRequired("Offset", SS.Offset);
    IO.mapRequired("Size", SS.Size);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::Argument::StackSpan)

template<>
struct llvm::yaml::MappingTraits<FTL::ReturnValue> {
  static void mapping(IO &IO, FTL::ReturnValue &RV) {
    IO.mapRequired("Type", RV.Type);
    IO.mapRequired("Registers", RV.Registers);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::ReturnValue)

template<>
struct llvm::yaml::MappingTraits<FTL::Argument> {
  static void mapping(IO &IO, FTL::Argument &A) {
    IO.mapRequired("Type", A.Type);
    IO.mapRequired("Kind", A.Kind);
    IO.mapRequired("Registers", A.Registers);
    IO.mapOptional("Stack", A.Stack);
  }
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FTL::Argument)

template<>
struct llvm::yaml::MappingTraits<FTL> {
  static void mapping(IO &IO, FTL &L) {
    IO.mapRequired("Arguments", L.Arguments);
    IO.mapRequired("ReturnValues", L.ReturnValues);
    IO.mapRequired("CalleeSavedRegisters", L.CalleeSavedRegisters);
    IO.mapRequired("FinalStackOffset", L.FinalStackOffset);
  }
};

void FTL::dump() const {
  // TODO: accept an arbitrary stream
  serialize(dbg, *this);
}
