//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE ABI
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ABI/Definition.h"

BOOST_AUTO_TEST_SUITE(RegisterStateDeduction);

BOOST_AUTO_TEST_CASE(NoArguments) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);

  auto Arguments = ABI.enforceArgumentRegisterState({});
  revng_check(Arguments.empty());

  auto RValues = ABI.enforceReturnValueRegisterState({});
  revng_check(RValues.empty());
}

BOOST_AUTO_TEST_CASE(OneGPRegister) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &GPRRetValues = ABI.GeneralPurposeReturnValueRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();
  const auto &VRRetValues = ABI.VectorReturnValueRegisters();

  abi::Definition::RegisterSet Arguments{ GPRArguments[0] };
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_check(Arguments.contains(GPRArguments[0]));
  revng_check(not Arguments.contains(VRArguments[0]));

  abi::Definition::RegisterSet RValues{ GPRRetValues[0] };
  RValues = ABI.enforceReturnValueRegisterState(std::move(RValues));
  revng_check(Arguments.contains(GPRRetValues[0]));
  revng_check(not Arguments.contains(VRRetValues[0]));
}

BOOST_AUTO_TEST_CASE(GPRArgumentsAndVRReturnValue) {
  auto ABI = abi::Definition::get(model::ABI::Microsoft_x86_64);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &GPRRetValues = ABI.GeneralPurposeReturnValueRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();
  const auto &VRRetValues = ABI.VectorReturnValueRegisters();

  abi::Definition::RegisterSet Arguments{ GPRArguments[2] };
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_check(Arguments.contains(GPRArguments[0]));
  revng_check(Arguments.contains(GPRArguments[1]));
  revng_check(Arguments.contains(GPRArguments[2]));
  revng_check(not Arguments.contains(GPRArguments[3]));
  revng_check(not Arguments.contains(VRArguments[0]));
  revng_check(not Arguments.contains(VRArguments[1]));
  revng_check(not Arguments.contains(VRArguments[2]));
  revng_check(not Arguments.contains(VRArguments[3]));

  abi::Definition::RegisterSet RValues{ VRRetValues[0] };
  RValues = ABI.enforceReturnValueRegisterState(std::move(RValues));
  revng_check(RValues.contains(VRRetValues[0]));
  revng_check(not RValues.contains(VRRetValues[1]));
  revng_check(not RValues.contains(GPRRetValues[0]));
  revng_check(not RValues.contains(GPRRetValues[1]));
}

BOOST_AUTO_TEST_CASE(DeduceFirstArgument) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();

  abi::Definition::RegisterSet Arguments{ VRArguments[1] };
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_check(Arguments.contains(GPRArguments[0]));
  revng_check(not Arguments.contains(GPRArguments[1]));
  revng_check(not Arguments.contains(GPRArguments[2]));
  revng_check(not Arguments.contains(GPRArguments[3]));
  revng_check(not Arguments.contains(VRArguments[0]));
  revng_check(Arguments.contains(VRArguments[1]));
  revng_check(not Arguments.contains(VRArguments[2]));
  revng_check(not Arguments.contains(VRArguments[3]));
}

BOOST_AUTO_TEST_CASE(DisambiguationFail) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();

  abi::Definition::RegisterSet Arguments{ GPRArguments[0], VRArguments[0] };
  auto Result = ABI.tryDeducingArgumentRegisterState(std::move(Arguments));
  revng_check(!Result.has_value());
}

BOOST_AUTO_TEST_CASE(UndetectableFail) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();

  abi::Definition::RegisterSet Arguments{ VRArguments[1] };
  auto Result = ABI.tryDeducingArgumentRegisterState(std::move(Arguments));
  revng_check(!Result.has_value());
}

BOOST_AUTO_TEST_CASE(UndetectableCornerCase) {
  auto ABI = abi::Definition::get(model::ABI::SystemZ_s390x);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();

  abi::Definition::RegisterSet Arguments{ VRArguments[1] };
  auto Result = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_assert(Result.contains(GPRArguments[0]));
  revng_assert(not Result.contains(GPRArguments[1]));
  revng_assert(not Result.contains(VRArguments[0]));
  revng_assert(Result.contains(VRArguments[1]));
}

BOOST_AUTO_TEST_CASE(MixedRegisters) {
  auto ABI = abi::Definition::get(model::ABI::SystemV_x86_64);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();
  const auto &VRArguments = ABI.VectorArgumentRegisters();

  abi::Definition::RegisterSet Arguments{ GPRArguments[2], VRArguments[2] };
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_check(Arguments.contains(GPRArguments[0]));
  revng_check(Arguments.contains(GPRArguments[1]));
  revng_check(Arguments.contains(GPRArguments[2]));
  revng_check(not Arguments.contains(GPRArguments[3]));
  revng_check(Arguments.contains(VRArguments[0]));
  revng_check(Arguments.contains(VRArguments[1]));
  revng_check(Arguments.contains(VRArguments[2]));
  revng_check(not Arguments.contains(VRArguments[3]));
}

BOOST_AUTO_TEST_CASE(AllTheRegisters) {
  auto ABI = abi::Definition::get(model::ABI::SystemV_x86_64);
  const auto &GPRArguments = ABI.GeneralPurposeArgumentRegisters();

  abi::Definition::RegisterSet Arguments(GPRArguments.begin(),
                                         GPRArguments.end());
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  for (model::Register::Values Register : GPRArguments)
    revng_check(Arguments.contains(Register));
}

BOOST_AUTO_TEST_CASE(ForbiddenRegister) {
  auto ABI = abi::Definition::get(model::ABI::SystemV_x86_64);

  auto Register = model::Register::getLast<model::Architecture::x86_64>();
  abi::Definition::RegisterSet Arguments{ Register };
  Arguments = ABI.enforceArgumentRegisterState(std::move(Arguments));
  revng_check(not Arguments.contains(Register));
}

BOOST_AUTO_TEST_CASE(ForbiddenRegisterFail) {
  auto ABI = abi::Definition::get(model::ABI::SystemV_x86_64);

  auto Register = model::Register::getLast<model::Architecture::x86_64>();
  abi::Definition::RegisterSet Arguments{ Register };
  auto Result = ABI.tryDeducingArgumentRegisterState(std::move(Arguments));
  revng_check(not Result.has_value());
}

BOOST_AUTO_TEST_SUITE_END();
