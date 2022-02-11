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
struct Trait<model::ABI::AAPCS> {
  static constexpr auto ABI = model::ABI::AAPCS;

  static constexpr bool ArgumentsArePositionBased = false;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = true;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = true;
  static constexpr bool UsePointerToCopyForStackArguments = false;

  static constexpr size_t MaximumGPRsPerAggregateArgument = 4;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 1;
  static constexpr size_t MaximumGPRsPerScalarArgument = 4;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 4;

  static constexpr std::array GeneralPurposeArgumentRegisters = {
    model::Register::r0_arm,
    model::Register::r1_arm,
    model::Register::r2_arm,
    model::Register::r3_arm
  };
  static constexpr std::array GeneralPurposeReturnValueRegisters = {
    model::Register::r0_arm,
    model::Register::r1_arm,
    model::Register::r2_arm,
    model::Register::r3_arm
  };

  static constexpr std::array VectorArgumentRegisters = {
    model::Register::q0_arm,
    model::Register::q1_arm,
    model::Register::q2_arm,
    model::Register::q3_arm
  };
  static constexpr std::array VectorReturnValueRegisters = {
    model::Register::q0_arm,
    model::Register::q1_arm,
    model::Register::q2_arm,
    model::Register::q3_arm
  };

  static constexpr std::array CalleeSavedRegisters = {
    model::Register::r4_arm,  model::Register::r5_arm,
    model::Register::r6_arm,  model::Register::r7_arm,
    model::Register::r8_arm,  model::Register::r10_arm,
    model::Register::r11_arm, model::Register::r13_arm,
    model::Register::q4_arm,  model::Register::q5_arm,
    model::Register::q6_arm,  model::Register::q7_arm
  };

  static constexpr auto ReturnValueLocationRegister = model::Register::r0_arm;
};

} // namespace abi
