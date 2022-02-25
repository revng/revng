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
struct Trait<model::ABI::SystemV_MIPS_o32> {
  static constexpr auto ABI = model::ABI::SystemV_MIPS_o32;

  static constexpr bool ArgumentsArePositionBased = true;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = true;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = true;
  static constexpr bool UsePointerToCopyForStackArguments = false;

  static constexpr size_t MaximumGPRsPerAggregateArgument = 4;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 2;
  static constexpr size_t MaximumGPRsPerScalarArgument = 4;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 2;

  static constexpr std::array GeneralPurposeArgumentRegisters = {
    model::Register::a0_mips,
    model::Register::a1_mips,
    model::Register::a2_mips,
    model::Register::a3_mips
  };
  static constexpr std::array GeneralPurposeReturnValueRegisters = {
    model::Register::v0_mips,
    model::Register::v1_mips
  };

  static constexpr std::array VectorArgumentRegisters = {
    model::Register::f12_mips,
    model::Register::f13_mips,
    model::Register::f14_mips,
    model::Register::f15_mips
  };
  static constexpr std::array VectorReturnValueRegisters = {
    model::Register::f0_mips,
    model::Register::f1_mips,
    model::Register::f2_mips,
    model::Register::f3_mips
  };

  static constexpr std::array CalleeSavedRegisters = {
    model::Register::s0_mips,  model::Register::s1_mips,
    model::Register::s2_mips,  model::Register::s3_mips,
    model::Register::s4_mips,  model::Register::s5_mips,
    model::Register::s6_mips,  model::Register::s7_mips,
    model::Register::gp_mips,  model::Register::sp_mips,
    model::Register::fp_mips,  model::Register::f20_mips,
    model::Register::f21_mips, model::Register::f22_mips,
    model::Register::f23_mips, model::Register::f24_mips,
    model::Register::f25_mips, model::Register::f26_mips,
    model::Register::f27_mips, model::Register::f28_mips,
    model::Register::f29_mips, model::Register::f30_mips,
    model::Register::f31_mips
  };

  static constexpr auto ReturnValueLocationRegister = model::Register::v0_mips;

  static constexpr bool CalleeIsResponsibleForStackCleanup = false;
  static constexpr size_t StackAlignment = 4;
  static constexpr size_t MinimumStackArgumentSize = 4;
};

} // namespace abi
