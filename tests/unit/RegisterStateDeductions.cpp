//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE ABI
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "revng/ABI/RegisterStateDeductions.h"
#include "revng/ABI/Trait.h"

using namespace model;
using namespace abi::RegisterState;

BOOST_AUTO_TEST_SUITE(RegisterStateDeductionSupportingInfrastructure);

BOOST_AUTO_TEST_CASE(RegisterStateMapTest) {
  abi::RegisterState::Map Map(Architecture::aarch64);

  for (auto [Register, State] : Map) {
    State.IsUsedForPassingArguments = size_t(Register) % 2 ? Yes : No;
    State.IsUsedForReturningValues = size_t(Register) % 2 ? No : Yes;
  }

  for (auto [R, State] : Map) {
    revng_check(State.IsUsedForPassingArguments == size_t(R) % 2 ? Yes : No);
    revng_check(State.IsUsedForReturningValues == size_t(R) % 2 ? No : Yes);
  }
}

BOOST_AUTO_TEST_SUITE_END();

BOOST_AUTO_TEST_SUITE(PositionBasedRegisterStateDeduction);

BOOST_AUTO_TEST_CASE(DefaultMap) {
  abi::RegisterState::Map Map(Architecture::systemz);

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(R.has_value());

  for (auto [Register, State] : R.value()) {
    revng_check(State.IsUsedForPassingArguments != Invalid);
    revng_check(!isYesOrDead(State.IsUsedForPassingArguments));

    revng_check(State.IsUsedForReturningValues != Invalid);
    revng_check(!isYesOrDead(State.IsUsedForReturningValues));
  }
}

BOOST_AUTO_TEST_CASE(NoArguments) {
  abi::RegisterState::Map Map(Architecture::systemz);
  for (auto [Register, State] : Map)
    State = { No, No };

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(R.has_value());
  for (const auto [Register, State] : R.value()) {
    revng_check(State.IsUsedForPassingArguments == No);
    revng_check(State.IsUsedForReturningValues == No);
  }
}

BOOST_AUTO_TEST_CASE(OneGPRegister) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto GPRRetValues = AT::GeneralPurposeReturnValueRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;
  constexpr auto VRRetValues = AT::VectorReturnValueRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Yes;
  Map[GPRRetValues[0]].IsUsedForReturningValues = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(R.has_value());
  revng_check(R->at(GPRArguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(R->at(GPRRetValues[0]).IsUsedForReturningValues == Yes);
  revng_check(R->at(VRArguments[0]).IsUsedForPassingArguments == No);
  revng_check(R->at(VRRetValues[0]).IsUsedForReturningValues == No);
}

BOOST_AUTO_TEST_CASE(OneGPRegisterAndOneVRegisterWithVReturnValue) {
  abi::RegisterState::Map Map(Architecture::mips);

  using AT = abi::Trait<ABI::SystemV_MIPS_o32>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto GPRRetValues = AT::GeneralPurposeReturnValueRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;
  constexpr auto VRRetValues = AT::VectorReturnValueRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Yes;
  if (VRArguments.size() > 1)
    Map[VRArguments[1]].IsUsedForPassingArguments = Yes;
  if (VRRetValues.size() > 1)
    Map[VRRetValues[1]].IsUsedForReturningValues = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_MIPS_o32, true);
  revng_check(R.has_value());
  revng_check(R->at(GPRArguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(R->at(VRArguments[0]).IsUsedForPassingArguments == No);
  revng_check(R->at(GPRRetValues[0]).IsUsedForReturningValues == No);
  revng_check(R->at(VRRetValues[0]).IsUsedForReturningValues == YesOrDead);
  if (GPRArguments.size() > 1)
    revng_check(R->at(GPRArguments[1]).IsUsedForPassingArguments == No);
  if (VRArguments.size() > 1)
    revng_check(R->at(VRArguments[1]).IsUsedForPassingArguments == Yes);
  if (GPRRetValues.size() > 1)
    revng_check(R->at(GPRRetValues[1]).IsUsedForReturningValues == No);
  if (VRRetValues.size() > 1)
    revng_check(R->at(VRRetValues[1]).IsUsedForReturningValues == Yes);
}

BOOST_AUTO_TEST_CASE(DeduceFirstArgumentToBeGPR) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto GPRRetValues = AT::GeneralPurposeReturnValueRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;
  constexpr auto VRRetValues = AT::VectorReturnValueRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[1]].IsUsedForPassingArguments = Yes;
  if (GPRRetValues.size() > 1)
    Map[GPRRetValues[0]].IsUsedForReturningValues = Maybe;
  if (VRRetValues.size() > 1)
    Map[VRRetValues[1]].IsUsedForReturningValues = Yes;

  constexpr static auto YoD = YesOrDead;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(R.has_value());
  revng_check(R->at(GPRArguments[0]).IsUsedForPassingArguments == YoD);
  if (GPRRetValues.size() > 1)
    revng_check(R->at(GPRRetValues[0]).IsUsedForReturningValues == No);
  revng_check(R->at(VRArguments[0]).IsUsedForPassingArguments == No);
  if (VRRetValues.size() > 1)
    revng_check(R->at(VRRetValues[0]).IsUsedForReturningValues == YoD);
  if (GPRArguments.size() > 1)
    revng_check(R->at(GPRArguments[1]).IsUsedForPassingArguments == No);
  if (VRArguments.size() > 1)
    revng_check(R->at(VRArguments[1]).IsUsedForPassingArguments == Yes);
  if (GPRRetValues.size() > 1)
    revng_check(R->at(GPRRetValues[1]).IsUsedForReturningValues == No);
  if (VRRetValues.size() > 1)
    revng_check(R->at(VRRetValues[1]).IsUsedForReturningValues == Yes);
}

BOOST_AUTO_TEST_CASE(DisambiguationFail) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Yes;
  Map[VRArguments[0]].IsUsedForPassingArguments = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(!R.has_value());
}

BOOST_AUTO_TEST_CASE(UndetectableFail) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[1]].IsUsedForPassingArguments = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemZ_s390x, true);
  revng_check(!R.has_value());
}

BOOST_AUTO_TEST_SUITE_END();

BOOST_AUTO_TEST_SUITE(NonPositionBasedRegisterStateDeduction);

BOOST_AUTO_TEST_CASE(DefaultMap) {
  abi::RegisterState::Map Map(Architecture::x86_64);

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  revng_check(R.has_value());

  for (auto [Register, State] : R.value()) {
    revng_check(State.IsUsedForPassingArguments != Invalid);
    revng_check(!isYesOrDead(State.IsUsedForPassingArguments));

    revng_check(State.IsUsedForReturningValues != Invalid);
    revng_check(!isYesOrDead(State.IsUsedForReturningValues));
  }
}

BOOST_AUTO_TEST_CASE(NoArguments) {
  abi::RegisterState::Map Map(Architecture::x86_64);
  for (auto [Register, State] : Map)
    State = { No, No };

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  revng_check(R.has_value());
  for (auto [Register, State] : R.value()) {
    revng_check(State.IsUsedForPassingArguments == No);
    revng_check(State.IsUsedForReturningValues == No);
  }
}

BOOST_AUTO_TEST_CASE(OneRegister) {
  abi::RegisterState::Map Map(Architecture::x86_64);
  for (auto [Register, State] : Map)
    State = { No, No };

  using AT = abi::Trait<ABI::SystemV_x86_64>;
  constexpr auto Arguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto RetValues = AT::GeneralPurposeReturnValueRegisters;

  Map[Arguments[0]].IsUsedForPassingArguments = Yes;
  Map[RetValues[0]].IsUsedForReturningValues = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  revng_check(R.has_value());
  revng_check(R->at(Arguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(R->at(RetValues[0]).IsUsedForReturningValues == Yes);
  revng_check(R->at(Arguments[1]).IsUsedForPassingArguments == No);
  revng_check(R->at(RetValues[1]).IsUsedForReturningValues == No);
}

BOOST_AUTO_TEST_CASE(TwoRegisters) {
  abi::RegisterState::Map Map(Architecture::x86_64);
  for (auto [Register, State] : Map)
    State = { No, No };

  using AT = abi::Trait<ABI::SystemV_x86_64>;
  constexpr auto Arguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto RetValues = AT::GeneralPurposeReturnValueRegisters;

  Map[Arguments[1]].IsUsedForPassingArguments = Yes;
  Map[RetValues[1]].IsUsedForReturningValues = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  revng_check(R.has_value());
  for (size_t Index = 0; Index < 2; ++Index) {
    auto MaybeA = R->at(Arguments[Index]).IsUsedForPassingArguments;
    revng_check(MaybeA != Invalid);
    revng_check(isYesOrDead(MaybeA));

    auto MaybeRV = R->at(RetValues[Index]).IsUsedForReturningValues;
    revng_check(MaybeRV != Invalid);
    revng_check(isYesOrDead(MaybeRV));
  }
  revng_check(R->at(Arguments[3]).IsUsedForPassingArguments == No);
  if constexpr (RetValues.size() > 2)
    revng_check(R->at(RetValues[3]).IsUsedForReturningValues == No);
}

BOOST_AUTO_TEST_CASE(AllRegisters) {
  abi::RegisterState::Map Map(Architecture::x86_64);
  for (auto [Register, State] : Map)
    State = { No, No };

  using AT = abi::Trait<ABI::SystemV_x86_64>;
  constexpr auto Arguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto RetValues = AT::GeneralPurposeReturnValueRegisters;

  Map[Arguments.back()].IsUsedForPassingArguments = Yes;
  Map[RetValues.back()].IsUsedForReturningValues = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  for (auto [Register, State] : R.value()) {
    if (llvm::is_contained(Arguments, Register)) {
      auto MaybeA = State.IsUsedForPassingArguments;
      revng_check(MaybeA != Invalid);
      revng_check(isYesOrDead(MaybeA));
    } else {
      revng_check(State.IsUsedForPassingArguments == No);
    }

    if (llvm::is_contained(RetValues, Register)) {
      auto MaybeRV = State.IsUsedForReturningValues;
      revng_check(MaybeRV != Invalid);
      revng_check(isYesOrDead(MaybeRV));
    } else {
      revng_check(State.IsUsedForReturningValues == No);
    }
  }
}

BOOST_AUTO_TEST_CASE(ForbiddenRegister) {
  abi::RegisterState::Map Map(Architecture::x86_64);

  static constexpr auto Register = Register::getLast<Architecture::x86_64>();
  Map[Register].IsUsedForPassingArguments = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, true);
  revng_check(R.has_value());
  revng_check(R->at(Register).IsUsedForPassingArguments == No);
}

BOOST_AUTO_TEST_CASE(ForbiddenRegisterFail) {
  abi::RegisterState::Map Map(Architecture::x86_64);

  Map[model::Register::fs_x86_64].IsUsedForPassingArguments = Yes;

  auto R = abi::applyRegisterStateDeductions(Map, ABI::SystemV_x86_64, false);
  revng_check(!R.has_value());
}

BOOST_AUTO_TEST_SUITE_END();
