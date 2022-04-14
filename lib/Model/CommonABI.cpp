/// \file CommonABIs.cpp
/// \brief A collection of typical ABIs to be accessible to the user.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/ABIDefinition.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/CommonABI.h"

namespace definitions {

static const model::ABIDefinition AAPCS64{
  /* .Architecture = */ model::Architecture::aarch64,
  /* .Name = */ "AAPCS64",

  /* .ArgumentsArePositionBased = */ false,
  /* .OnlyStartDoubleArgumentsFromAnEvenRegister = */ true,
  /* .ArgumentsCanBeSplitBetweenRegistersAndStack = */ true,
  /* .UsePointerToCopyForStackArguments = */ false,

  /* .MaximumGPRsPerAggregateArgument = */ 2,
  /* .MaximumGPRsPerAggregateReturnValue = */ 2,
  /* .MaximumGPRsPerScalarArgument = */ 2,
  /* .MaximumGPRsPerScalarReturnValue = */ 2,

  /* .GeneralPurposeArgumentRegisters = */
  { model::Register::x0_aarch64,
    model::Register::x1_aarch64,
    model::Register::x2_aarch64,
    model::Register::x3_aarch64,
    model::Register::x4_aarch64,
    model::Register::x5_aarch64,
    model::Register::x6_aarch64,
    model::Register::x7_aarch64 },

  /* .GeneralPurposeReturnValueRegisters = */
  { model::Register::x0_aarch64,
    model::Register::x1_aarch64,
    model::Register::x2_aarch64,
    model::Register::x3_aarch64,
    model::Register::x4_aarch64,
    model::Register::x5_aarch64,
    model::Register::x6_aarch64,
    model::Register::x7_aarch64 },

  /* .VectorArgumentRegisters = */
  { model::Register::v0_aarch64,
    model::Register::v1_aarch64,
    model::Register::v2_aarch64,
    model::Register::v3_aarch64,
    model::Register::v4_aarch64,
    model::Register::v5_aarch64,
    model::Register::v6_aarch64,
    model::Register::v7_aarch64 },

  /* .VectorReturnValueRegisters = */
  { model::Register::v0_aarch64,
    model::Register::v1_aarch64,
    model::Register::v2_aarch64,
    model::Register::v3_aarch64,
    model::Register::v4_aarch64,
    model::Register::v5_aarch64,
    model::Register::v6_aarch64,
    model::Register::v7_aarch64 },

  /* .CalleeSavedRegisters = */
  { model::Register::x19_aarch64, model::Register::x20_aarch64,
    model::Register::x21_aarch64, model::Register::x22_aarch64,
    model::Register::x23_aarch64, model::Register::x24_aarch64,
    model::Register::x25_aarch64, model::Register::x26_aarch64,
    model::Register::x27_aarch64, model::Register::x28_aarch64,
    model::Register::x29_aarch64, model::Register::sp_aarch64,
    model::Register::v8_aarch64,  model::Register::v9_aarch64,
    model::Register::v10_aarch64, model::Register::v11_aarch64,
    model::Register::v12_aarch64, model::Register::v13_aarch64,
    model::Register::v14_aarch64, model::Register::v15_aarch64 },

  /* .ReturnValueLocationRegister = */ model::Register::x8_aarch64,
  /* .CalleeIsResponsibleForStackCleanup = */ false,
  /* .StackAlignment = */ 16,
  /* .MinimumStackArgumentSize = */ 8
};

} // namespace definitions

const model::ABIDefinition &
model::CommonABI::define(model::CommonABI::Values V) {
  switch (V) {
  case model::CommonABI::AAPCS64:
    return definitions::AAPCS64;
  default:
    revng_abort("TODO: Define all the other common ABIs");
  }
}
