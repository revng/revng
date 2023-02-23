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
struct Trait<model::ABI::SystemV_x86_64> {
  static constexpr auto ABI = model::ABI::SystemV_x86_64;

  static constexpr bool ArgumentsArePositionBased = false;
  static constexpr bool OnlyStartDoubleArgumentsFromAnEvenRegister = false;
  static constexpr bool ArgumentsCanBeSplitBetweenRegistersAndStack = false;
  static constexpr bool UsePointerToCopyForStackArguments = false;

  /// \note: if the aggregate argument contains SSE-fields, this doesn't apply.
  static constexpr size_t MaximumGPRsPerAggregateArgument = 2;
  static constexpr size_t MaximumGPRsPerAggregateReturnValue = 2;
  static constexpr size_t MaximumGPRsPerScalarArgument = 8;
  static constexpr size_t MaximumGPRsPerScalarReturnValue = 2;

  static constexpr std::array GeneralPurposeArgumentRegisters = {
    model::Register::rdi_x86_64, model::Register::rsi_x86_64,
    model::Register::rdx_x86_64, model::Register::rcx_x86_64,
    model::Register::r8_x86_64,  model::Register::r9_x86_64
  };
  static constexpr std::array GeneralPurposeReturnValueRegisters = {
    model::Register::rax_x86_64,
    model::Register::rdx_x86_64
  };

  static constexpr std::array VectorArgumentRegisters = {
    model::Register::xmm0_x86_64, model::Register::xmm1_x86_64,
    model::Register::xmm2_x86_64, model::Register::xmm3_x86_64,
    model::Register::xmm4_x86_64, model::Register::xmm5_x86_64,
    model::Register::xmm6_x86_64, model::Register::xmm7_x86_64
  };
  static constexpr std::array VectorReturnValueRegisters = {
    model::Register::xmm0_x86_64,
    model::Register::xmm1_x86_64
  };

  static constexpr std::array CalleeSavedRegisters = {
    model::Register::rbx_x86_64, model::Register::rbp_x86_64,
    model::Register::rsp_x86_64, model::Register::r12_x86_64,
    model::Register::r13_x86_64, model::Register::r14_x86_64,
    model::Register::r15_x86_64
  };

  static constexpr auto
    ReturnValueLocationRegister = model::Register::rdi_x86_64;

  static constexpr bool CalleeIsResponsibleForStackCleanup = false;
  static constexpr size_t StackAlignment = 16;
};

} // namespace abi
