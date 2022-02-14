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
struct Trait<model::ABI::Microsoft_x86_clrcall> {
  static constexpr auto ABI = model::ABI::Microsoft_x86_clrcall;

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
    model::Register::ebx_x86,
    model::Register::ebp_x86,
    model::Register::esp_x86,
    model::Register::edi_x86,
    model::Register::esi_x86
  };

  static constexpr auto ReturnValueLocationRegister = model::Register::eax_x86;
};

} // namespace abi
