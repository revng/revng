#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/ABI.h"
#include "revng/Model/Register.h"

namespace abi {

template<model::ABI::Values ABI>
struct Trait;

template<>
struct Trait<model::ABI::SystemZ_s390x> {
  static constexpr auto ABI = model::ABI::SystemZ_s390x;

  static constexpr bool ArgumentsArePositionBased = true;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = false;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = false;
  static constexpr bool UsePointerToCopyForStackArguments = true;

  static constexpr size_t MaximumGPRsPerAggregateArgument = 1;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 1;
  static constexpr size_t MaximumGPRsPerScalarArgument = 1;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 1;

  static constexpr std::array GeneralPurposeArgumentRegisters = {
    model::Register::r2_systemz,
    model::Register::r3_systemz,
    model::Register::r4_systemz,
    model::Register::r5_systemz,
    model::Register::r6_systemz
  };
  static constexpr std::array GeneralPurposeReturnValueRegisters = {
    model::Register::r2_systemz
  };

  static constexpr std::array VectorArgumentRegisters = {
    model::Register::f0_systemz,
    model::Register::f1_systemz,
    model::Register::f2_systemz,
    model::Register::f3_systemz,
    model::Register::f4_systemz,
    model::Register::f5_systemz,
    model::Register::f6_systemz,
    model::Register::f7_systemz
    // model::Register::v24_systemz,
    // model::Register::v26_systemz,
    // model::Register::v28_systemz,
    // model::Register::v30_systemz,
    // model::Register::v25_systemz,
    // model::Register::v27_systemz,
    // model::Register::v29_systemz,
    // model::Register::v31_systemz
  };
  static constexpr std::array VectorReturnValueRegisters = {
    model::Register::f0_systemz
    // model::Register::v24_systemz,
  };

  static constexpr std::array CalleeSavedRegisters = {
    model::Register::r6_systemz,  model::Register::r7_systemz,
    model::Register::r8_systemz,  model::Register::r9_systemz,
    model::Register::r10_systemz, model::Register::r11_systemz,
    model::Register::r12_systemz, model::Register::r13_systemz,
    model::Register::r15_systemz, model::Register::f8_systemz,
    model::Register::f9_systemz,  model::Register::f10_systemz,
    model::Register::f11_systemz, model::Register::f12_systemz,
    model::Register::f13_systemz, model::Register::f15_systemz
  };

  static constexpr auto
    ReturnValueLocationRegister = model::Register::r2_systemz;

  static constexpr bool CalleeIsResponsibleForStackCleanup = false;
  static constexpr size_t StackAlignment = 8;
};

} // namespace abi
