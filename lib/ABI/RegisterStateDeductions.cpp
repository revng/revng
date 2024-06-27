/// \file RegisterStateDeductions.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <span>

#include "llvm/ADT/ArrayRef.h"

#include "revng/ABI/Definition.h"
#include "revng/Model/Binary.h"

static Logger Log("abi-register-state-deduction");

/// Finds last relevant (`Yes` or `Dead`) register out of the provided list.
///
/// Returns `0` if no registers from the list were mentioned.
static size_t
findLastUsedIndex(llvm::ArrayRef<model::Register::Values> Registers,
                  const abi::Definition::RegisterSet &State) {
  for (size_t Index = Registers.size(); Index != 0; --Index)
    if (State.contains(Registers[Index - 1]))
      return Index;

  return 0;
}

using Def = abi::Definition;

template<bool EnforceABIConformance>
struct DeductionImpl {
  const abi::Definition &ABI;
  const std::string_view ABIName;
  explicit DeductionImpl(const abi::Definition &ABI) :
    ABI(ABI), ABIName(model::ABI::getName(ABI.ABI())) {}

  std::optional<Def::RegisterSet> arguments(Def::RegisterSet Arguments) {
    if (!ensureRegistersAreAllowed(Arguments, allowedArgumentRegisters()))
      return std::nullopt;

    if (ABI.ArgumentsArePositionBased()) {
      if (deducePositionBasedArguments(Arguments))
        return std::move(Arguments);

    } else {
      if (deduceNonPositionBasedArguments(Arguments))
        return std::move(Arguments);
    }

    return std::nullopt;
  }

  std::optional<Def::RegisterSet> returnValues(Def::RegisterSet ReturnValues) {
    if (!ensureRegistersAreAllowed(ReturnValues, allowedReturnValueRegisters()))
      return std::nullopt;

    if (deduceReturnValues(ReturnValues))
      return std::move(ReturnValues);

    return std::nullopt;
  }

private:
  using CRegister = const model::Register::Values;
  auto allowedArgumentRegisters() const {
    return llvm::concat<CRegister>(ABI.GeneralPurposeArgumentRegisters(),
                                   ABI.VectorArgumentRegisters());
  }
  auto allowedReturnValueRegisters() const {
    return llvm::concat<CRegister>(ABI.GeneralPurposeReturnValueRegisters(),
                                   ABI.VectorReturnValueRegisters());
  }

  template<range_with_value_type<CRegister> Registers>
  bool ensureRegistersAreAllowed(Def::RegisterSet &UsedSet,
                                 Registers &&AllowedRegisters) const {
    if constexpr (EnforceABIConformance == true) {
      size_t CountBefore = UsedSet.size();
      std::erase_if(UsedSet, [&](model::Register::Values Register) {
        return !llvm::is_contained(AllowedRegisters, Register);
      });
      if (size_t RemovedCount = UsedSet.size() != CountBefore)
        revng_log(Log,
                  "Removing " << RemovedCount
                              << " registers from the set, as ABI doesn't "
                                 "allow them to be used.");

    } else {
      for (model::Register::Values Register : UsedSet) {
        if (!llvm::is_contained(AllowedRegisters, Register)) {
          revng_log(Log,
                    "Aborting, `model::Register::"
                      << model::Register::getName(Register).data()
                      << "` register is used despite not being allowed by `"
                      << ABIName << "` ABI.");
          return false;
        }
      }
    }

    return true;
  }

  bool deducePositionBasedArguments(Def::RegisterSet &State) {
    llvm::ArrayRef GPAR = ABI.GeneralPurposeArgumentRegisters();
    llvm::ArrayRef VAR = ABI.VectorArgumentRegisters();

    bool IsRequired = false;
    if (GPAR.size() > VAR.size()) {
      for (auto Register : llvm::reverse(GPAR.drop_front(VAR.size()))) {
        if (State.contains(Register))
          IsRequired = true;
        else if (IsRequired)
          State.emplace(Register);
      }
      GPAR = GPAR.take_front(VAR.size());
    } else if (VAR.size() > GPAR.size()) {
      for (auto Register : llvm::reverse(VAR.drop_front(GPAR.size()))) {
        if (State.contains(Register))
          IsRequired = true;
        else if (IsRequired)
          State.emplace(Register);
      }
      VAR = VAR.take_front(GPAR.size());
    }

    auto ArgumentRange = llvm::zip(llvm::reverse(GPAR), llvm::reverse(VAR));
    for (auto [GPR, VR] : ArgumentRange)
      if (!singlePositionBasedDeduction(GPR, VR, State, IsRequired))
        return false;

    return true;
  }

  bool singlePositionBasedDeduction(model::Register::Values GPRegister,
                                    model::Register::Values VRegister,
                                    Def::RegisterSet &State,
                                    bool &IsRequired) {
    if (State.contains(GPRegister)) {
      if (State.contains(VRegister)) {
        // Both are set - there's no way to tell which one is actually used.
        if constexpr (EnforceABIConformance == true) {
          // Pick one arbitrarily because most likely both of them are `Dead`.
          State.erase(VRegister);
        } else {
          // Report the problem and abort.
          revng_log(Log,
                    "Impossible to differentiate which one of the two "
                    "registers (`model::Register::"
                      << model::Register::getName(GPRegister).data()
                      << "` and `model::Register::"
                      << model::Register::getName(VRegister).data()
                      << "`) should be used: both are `YesOrDead`. The ABI is "
                      << ABIName << ".");
          return false;
        }
      } else {
        // Only general purpose one is set: we're happy.
        IsRequired = true;
      }
    } else {
      if (State.contains(VRegister)) {
        // Only vector one is set: we're happy.
        IsRequired = true;
      } else {
        // Neither one is set.
        if (IsRequired) {
          if constexpr (EnforceABIConformance == true) {
            // Pick one arbitrarily because most likely both of them are `Dead`.
            State.emplace(GPRegister);
          } else {
            // Report the problem and abort.
            revng_log(Log,
                      "Impossible to differentiate which one of the two "
                      "registers (`model::Register::"
                        << model::Register::getName(GPRegister).data()
                        << "` and `model::Register::"
                        << model::Register::getName(VRegister).data()
                        << "`) should be used: neither is `YesOrDead`. "
                        << "The ABI is " << ABIName << ".");
            return false;
          }
        }
      }
    }

    return true;
  }

  template<range_with_value_type<CRegister> Registers>
  void fillState(Def::RegisterSet &State, Registers RequiredRegisters) {
    for (model::Register::Values Register : RequiredRegisters)
      State.emplace(Register);
  }

  bool deduceNonPositionBasedArguments(Def::RegisterSet &State) {
    llvm::ArrayRef GPs = ABI.GeneralPurposeArgumentRegisters();
    llvm::ArrayRef RequiredGPRs = GPs.take_front(findLastUsedIndex(GPs, State));
    llvm::ArrayRef VRs = ABI.VectorArgumentRegisters();
    llvm::ArrayRef RequiredVRs = VRs.take_front(findLastUsedIndex(VRs, State));

    fillState(State, llvm::concat<CRegister>(RequiredGPRs, RequiredVRs));

    return true;
  }

  bool deduceReturnValues(Def::RegisterSet &State) {
    llvm::ArrayRef GPs = ABI.GeneralPurposeReturnValueRegisters();
    llvm::ArrayRef RequiredGPRs = GPs.take_front(findLastUsedIndex(GPs, State));
    llvm::ArrayRef VRs = ABI.VectorReturnValueRegisters();
    llvm::ArrayRef RequiredVRs = VRs.take_front(findLastUsedIndex(VRs, State));

    if (!RequiredGPRs.empty()) {
      if (!RequiredVRs.empty()) {
        if constexpr (EnforceABIConformance == true) {
          // Even though we shouldn't let both through, we have no way
          // to differentiate between the two. So just keep them as is.
        } else {
          revng_log(Log,
                    "Impossible to differentiate whether the return value uses "
                    "general purpose registers or in vector ones: it cannot be "
                    "both. The ABI is "
                      << ABIName << ".");
          return false;
        }
      } else {
        // Only general purpose registers are used: we're happy.
        fillState(State, RequiredGPRs);
      }
    } else {
      if (!RequiredVRs.empty()) {
        // Only vector registers are used: we're happy.
        fillState(State, RequiredVRs);
      } else {
        // No return value (???)
        // Since there's no way to confirm, do nothing.
      }
    }

    return true;
  }
};

using SoftDeduction = DeductionImpl<false>;
using StrictDeduction = DeductionImpl<true>;

std::optional<abi::Definition::RegisterSet>
Def::tryDeducingArgumentRegisterState(RegisterSet &&Arguments) const {
  return SoftDeduction(*this).arguments(std::move(Arguments));
}

std::optional<abi::Definition::RegisterSet>
Def::tryDeducingReturnValueRegisterState(RegisterSet &&ReturnValues) const {
  return SoftDeduction(*this).returnValues(std::move(ReturnValues));
}

abi::Definition::RegisterSet
Def::enforceArgumentRegisterState(RegisterSet &&Arguments) const {
  auto Result = StrictDeduction(*this).arguments(std::move(Arguments));
  revng_assert(Result != std::nullopt);
  return Result.value();
}

abi::Definition::RegisterSet
Def::enforceReturnValueRegisterState(RegisterSet &&ReturnValues) const {
  auto Result = StrictDeduction(*this).returnValues(std::move(ReturnValues));
  revng_assert(Result != std::nullopt);
  return Result.value();
}
