#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"

#include "revng/ABI/Trait.h"
#include "revng/Model/ABI.h"
#include "revng/Support/EnumSwitch.h"

namespace abi {

constexpr bool areArgumentsPositionBased(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::ArgumentsArePositionBased;
  });
}

constexpr bool
canOnlyStartDoubleArgumentsFromAnEvenRegister(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::OnlyStartDoubleArgumentsFromAnEvenRegister;
  });
}

constexpr bool
canArgumentsBeSplitBetweenRegistersAndStack(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::ArgumentsCanBeSplitBetweenRegistersAndStack;
  });
}

constexpr bool
canOnlyUsePointerToCopyForStackArguments(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::UsePointerToCopyForStackArguments;
  });
}

constexpr size_t countMaximumGPRsPerAggregateArgument(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::MaximumGPRsPerAggregateArgument;
  });
}

constexpr size_t
countMaximumGPRsPerAggregateReturnValue(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::MaximumGPRsPerAggregateReturnValue;
  });
}

constexpr size_t countMaximumGPRsPerScalarArgument(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::MaximumGPRsPerScalarArgument;
  });
}

constexpr size_t countMaximumGPRsPerScalarReturnValue(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::MaximumGPRsPerScalarReturnValue;
  });
}

constexpr llvm::ArrayRef<model::Register::Values>
listGeneralPurposeArgumentRegisters(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  using Registers = llvm::ArrayRef<model::Register::Values>;
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() -> Registers {
    return abi::Trait<A>::GeneralPurposeArgumentRegisters;
  });
}

constexpr llvm::ArrayRef<model::Register::Values>
listGeneralPurposeReturnValueRegisters(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  using Registers = llvm::ArrayRef<model::Register::Values>;
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() -> Registers {
    return abi::Trait<A>::GeneralPurposeReturnValueRegisters;
  });
}

constexpr llvm::ArrayRef<model::Register::Values>
listVectorArgumentRegisters(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  using Registers = llvm::ArrayRef<model::Register::Values>;
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() -> Registers {
    return abi::Trait<A>::VectorArgumentRegisters;
  });
}

constexpr llvm::ArrayRef<model::Register::Values>
listVectorReturnValueRegisters(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  using Registers = llvm::ArrayRef<model::Register::Values>;
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() -> Registers {
    return abi::Trait<A>::VectorReturnValueRegisters;
  });
}

constexpr llvm::ArrayRef<model::Register::Values>
listCalleeSavedRegisters(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  using Registers = llvm::ArrayRef<model::Register::Values>;
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() -> Registers {
    return abi::Trait<A>::CalleeSavedRegisters;
  });
}

constexpr model::Register::Values
getReturnValueLocationRegister(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::ReturnValueLocationRegister;
  });
}

constexpr bool isCalleeResponsibleForStackCleanup(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::CalleeIsResponsibleForStackCleanup;
  });
}

constexpr size_t getStackAlignment(model::ABI::Values ABI) {
  revng_assert(ABI != model::ABI::Invalid);
  return skippingEnumSwitch<1>(ABI, [&]<model::ABI::Values A>() {
    return abi::Trait<A>::StackAlignment;
  });
}

} // namespace abi
