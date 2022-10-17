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
struct Trait<model::ABI::AAPCS64> {
  static constexpr auto ABI = model::ABI::AAPCS64;

  static constexpr bool ArgumentsArePositionBased = false;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = true;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = true;
  static constexpr bool UsePointerToCopyForStackArguments = false;

  static constexpr size_t MaximumGPRsPerAggregateArgument = 2;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 2;
  static constexpr size_t MaximumGPRsPerScalarArgument = 2;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 2;

  static constexpr std::array GeneralPurposeArgumentRegisters = {
    model::Register::x0_aarch64, model::Register::x1_aarch64,
    model::Register::x2_aarch64, model::Register::x3_aarch64,
    model::Register::x4_aarch64, model::Register::x5_aarch64,
    model::Register::x6_aarch64, model::Register::x7_aarch64
  };
  static constexpr std::array GeneralPurposeReturnValueRegisters = {
    model::Register::x0_aarch64, model::Register::x1_aarch64,
    model::Register::x2_aarch64, model::Register::x3_aarch64,
    model::Register::x4_aarch64, model::Register::x5_aarch64,
    model::Register::x6_aarch64, model::Register::x7_aarch64
  };

  static constexpr std::array VectorArgumentRegisters = {
    model::Register::v0_aarch64, model::Register::v1_aarch64,
    model::Register::v2_aarch64, model::Register::v3_aarch64,
    model::Register::v4_aarch64, model::Register::v5_aarch64,
    model::Register::v6_aarch64, model::Register::v7_aarch64
  };
  static constexpr std::array VectorReturnValueRegisters = {
    model::Register::v0_aarch64, model::Register::v1_aarch64,
    model::Register::v2_aarch64, model::Register::v3_aarch64,
    model::Register::v4_aarch64, model::Register::v5_aarch64,
    model::Register::v6_aarch64, model::Register::v7_aarch64
  };

  static constexpr std::array CalleeSavedRegisters = {
    model::Register::x19_aarch64, model::Register::x20_aarch64,
    model::Register::x21_aarch64, model::Register::x22_aarch64,
    model::Register::x23_aarch64, model::Register::x24_aarch64,
    model::Register::x25_aarch64, model::Register::x26_aarch64,
    model::Register::x27_aarch64, model::Register::x28_aarch64,
    model::Register::x29_aarch64, model::Register::sp_aarch64,
    model::Register::v8_aarch64,  model::Register::v9_aarch64,
    model::Register::v10_aarch64, model::Register::v11_aarch64,
    model::Register::v12_aarch64, model::Register::v13_aarch64,
    model::Register::v14_aarch64, model::Register::v15_aarch64
  };

  static constexpr auto
    ReturnValueLocationRegister = model::Register::x8_aarch64;

  static constexpr bool CalleeIsResponsibleForStackCleanup = false;
  static constexpr size_t StackAlignment = 16;
};

} // namespace abi
