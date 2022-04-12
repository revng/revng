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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  for (auto [Register, State] : Result) {
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  for (const auto [Register, State] : Result) {
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  revng_check(Result.at(GPRArguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(Result.at(GPRRetValues[0]).IsUsedForReturningValues == Yes);
  revng_check(Result.at(VRArguments[0]).IsUsedForPassingArguments == No);
  revng_check(Result.at(VRRetValues[0]).IsUsedForReturningValues == No);
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_MIPS_o32);
  revng_check(Result.at(GPRArguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(Result.at(VRArguments[0]).IsUsedForPassingArguments == No);
  revng_check(Result.at(GPRRetValues[0]).IsUsedForReturningValues == No);
  revng_check(Result.at(VRRetValues[0]).IsUsedForReturningValues == YesOrDead);
  if (GPRArguments.size() > 1)
    revng_check(Result.at(GPRArguments[1]).IsUsedForPassingArguments == No);
  if (VRArguments.size() > 1)
    revng_check(Result.at(VRArguments[1]).IsUsedForPassingArguments == Yes);
  if (GPRRetValues.size() > 1)
    revng_check(Result.at(GPRRetValues[1]).IsUsedForReturningValues == No);
  if (VRRetValues.size() > 1)
    revng_check(Result.at(VRRetValues[1]).IsUsedForReturningValues == Yes);
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  revng_check(Result.at(GPRArguments[0]).IsUsedForPassingArguments == YoD);
  if (GPRRetValues.size() > 1)
    revng_check(Result.at(GPRRetValues[0]).IsUsedForReturningValues == No);
  revng_check(Result.at(VRArguments[0]).IsUsedForPassingArguments == No);
  if (VRRetValues.size() > 1)
    revng_check(Result.at(VRRetValues[0]).IsUsedForReturningValues == YoD);
  if (GPRArguments.size() > 1)
    revng_check(Result.at(GPRArguments[1]).IsUsedForPassingArguments == No);
  if (VRArguments.size() > 1)
    revng_check(Result.at(VRArguments[1]).IsUsedForPassingArguments == Yes);
  if (GPRRetValues.size() > 1)
    revng_check(Result.at(GPRRetValues[1]).IsUsedForReturningValues == No);
  if (VRRetValues.size() > 1)
    revng_check(Result.at(VRRetValues[1]).IsUsedForReturningValues == Yes);
}

BOOST_AUTO_TEST_CASE(DisambiguationFail) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Yes;
  Map[VRArguments[0]].IsUsedForPassingArguments = Yes;

  auto Result = abi::tryApplyRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  revng_check(!Result.has_value());
}

BOOST_AUTO_TEST_CASE(UndetectableFail) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[1]].IsUsedForPassingArguments = Yes;

  auto Result = abi::tryApplyRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  revng_check(!Result.has_value());
}

BOOST_AUTO_TEST_CASE(UndetectableCornerCase) {
  abi::RegisterState::Map Map(Architecture::systemz);

  using AT = abi::Trait<ABI::SystemZ_s390x>;
  constexpr auto GPRArguments = AT::GeneralPurposeArgumentRegisters;
  constexpr auto VRArguments = AT::VectorArgumentRegisters;

  Map[GPRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[0]].IsUsedForPassingArguments = Maybe;
  Map[VRArguments[1]].IsUsedForPassingArguments = Yes;

  constexpr static auto YoD = YesOrDead;

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemZ_s390x);
  revng_check(Result.at(GPRArguments[0]).IsUsedForPassingArguments == YoD);
  revng_check(Result.at(VRArguments[0]).IsUsedForPassingArguments == No);
}

BOOST_AUTO_TEST_SUITE_END();

BOOST_AUTO_TEST_SUITE(NonPositionBasedRegisterStateDeduction);

BOOST_AUTO_TEST_CASE(DefaultMap) {
  abi::RegisterState::Map Map(Architecture::x86_64);

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  for (auto [Register, State] : Result) {
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  for (auto [Register, State] : Result) {
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  revng_check(Result.at(Arguments[0]).IsUsedForPassingArguments == Yes);
  revng_check(Result.at(RetValues[0]).IsUsedForReturningValues == Yes);
  revng_check(Result.at(Arguments[1]).IsUsedForPassingArguments == No);
  revng_check(Result.at(RetValues[1]).IsUsedForReturningValues == No);
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  for (size_t Index = 0; Index < 2; ++Index) {
    auto MaybeA = Result.at(Arguments[Index]).IsUsedForPassingArguments;
    revng_check(MaybeA != Invalid);
    revng_check(isYesOrDead(MaybeA));

    auto MaybeRV = Result.at(RetValues[Index]).IsUsedForReturningValues;
    revng_check(MaybeRV != Invalid);
    revng_check(isYesOrDead(MaybeRV));
  }
  revng_check(Result.at(Arguments[3]).IsUsedForPassingArguments == No);
  if constexpr (RetValues.size() > 2)
    revng_check(Result.at(RetValues[3]).IsUsedForReturningValues == No);
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  for (auto [Register, State] : Result) {
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

  auto Result = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  revng_check(Result.at(Register).IsUsedForPassingArguments == No);
}

BOOST_AUTO_TEST_CASE(ForbiddenRegisterFail) {
  abi::RegisterState::Map Map(Architecture::x86_64);

  Map[model::Register::fs_x86_64].IsUsedForPassingArguments = Yes;

  auto Result = abi::tryApplyRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  revng_check(!Result.has_value());
}

using RegisterState = abi::RegisterState::Values;
static bool checkDeductionHelper(RegisterState Input,
                                 std::optional<RegisterState> ExpectedOutput,
                                 RegisterState ExpectedEnforcedOutput) {
  abi::RegisterState::Map Map(Architecture::x86_64);
  Map[model::Register::rdi_x86_64].IsUsedForPassingArguments = Input;
  Map[model::Register::rsi_x86_64].IsUsedForPassingArguments = Yes;

  auto Tried = abi::tryApplyRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  if (!Tried.has_value()) {
    if (ExpectedOutput.has_value())
      return false;
  } else {
    const auto &RDIState = Tried->at(model::Register::rdi_x86_64);
    if (*ExpectedOutput != RDIState.IsUsedForPassingArguments)
      return false;
  }

  auto Enforced = abi::enforceRegisterStateDeductions(Map, ABI::SystemV_x86_64);
  const auto &RDIState = Enforced.at(model::Register::rdi_x86_64);
  if (ExpectedEnforcedOutput != RDIState.IsUsedForPassingArguments)
    return false;

  return true;
}

BOOST_AUTO_TEST_CASE(SingleNonPositionBasedDeductionTable) {
  revng_check(checkDeductionHelper(No, std::nullopt, YesOrDead));
  revng_check(checkDeductionHelper(NoOrDead, Dead, Dead));
  revng_check(checkDeductionHelper(Dead, Dead, Dead));
  revng_check(checkDeductionHelper(Yes, Yes, Yes));
  revng_check(checkDeductionHelper(YesOrDead, YesOrDead, YesOrDead));
  revng_check(checkDeductionHelper(Maybe, YesOrDead, YesOrDead));
  revng_check(checkDeductionHelper(Contradiction, std::nullopt, YesOrDead));
}

BOOST_AUTO_TEST_SUITE_END();
