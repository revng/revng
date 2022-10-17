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
struct Trait<model::ABI::Microsoft_x86_64_clrcall> {
  static constexpr auto ABI = model::ABI::Microsoft_x86_64_clrcall;

  static constexpr bool ArgumentsArePositionBased = false;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = false;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = false;
  static constexpr bool UsePointerToCopyForStackArguments = false;

  static constexpr size_t MaximumGPRsPerAggregateArgument = 0;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 0;
  static constexpr size_t MaximumGPRsPerScalarArgument = 0;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 0;

  static constexpr std::array<model::Register::Values, 0>
    GeneralPurposeArgumentRegisters = {};
  static constexpr std::array<model::Register::Values, 0>
    GeneralPurposeReturnValueRegisters = {};

  static constexpr std::array<model::Register::Values, 0>
    VectorArgumentRegisters = {};
  static constexpr std::array<model::Register::Values, 0>
    VectorReturnValueRegisters = {};

  /// \note: it's probably a good idea to double-check `CalleeSavedRegisters`,
  /// (for now it lists the defaults), but I'm not sure where I could find
  /// that information.
  static constexpr std::array CalleeSavedRegisters = {
    model::Register::r12_x86_64,  model::Register::r13_x86_64,
    model::Register::r14_x86_64,  model::Register::r15_x86_64,
    model::Register::rdi_x86_64,  model::Register::rsi_x86_64,
    model::Register::rbx_x86_64,  model::Register::rbp_x86_64,
    model::Register::xmm6_x86_64, model::Register::xmm7_x86_64
  };

  static constexpr auto ReturnValueLocationRegister = model::Register::Invalid;

  static constexpr bool CalleeIsResponsibleForStackCleanup = false;
  static constexpr size_t StackAlignment = 16;
};

} // namespace abi
