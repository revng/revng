/// \file ApplyRegisterStateDeductions.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <span>

#include "llvm/ADT/ArrayRef.h"

#include "revng/ABI/Definition.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Types.h"

static Logger Log("abi-register-state-deduction");

using State = abi::RegisterState::Values;
using StateMap = abi::RegisterState::Map;
using StatePair = StateMap::StatePair;
using AccessorType = State &(StatePair &);
using CAccessorType = const State &(const StatePair &);

static State &accessArgument(StatePair &Input) {
  return Input.IsUsedForPassingArguments;
}
static State &accessReturnValue(StatePair &Input) {
  return Input.IsUsedForReturningValues;
}

static const State &accessCArgument(const StatePair &Input) {
  return Input.IsUsedForPassingArguments;
}
static const State &accessCReturnValue(const StatePair &Input) {
  return Input.IsUsedForReturningValues;
}

using CRegister = const model::Register::Values;

/// Finds last relevant (`Yes` or `Dead`) register out of the provided list.
///
/// Returns `0` if no registers from the list were mentioned.
template<CAccessorType Accessor>
size_t findLastUsedIndex(llvm::ArrayRef<model::Register::Values> Registers,
                         const StateMap &State) {
  namespace RS = abi::RegisterState;
  for (size_t Index = Registers.size(); Index != 0; --Index)
    if (RS::isYesOrDead(Accessor(State.at(Registers[Index - 1]))))
      return Index;

  return 0;
}

template<bool EnforceABIConformance>
struct DeductionImpl {
  const abi::Definition &ABI;
  const std::string_view ABIName;
  explicit DeductionImpl(const abi::Definition &ABI) :
    ABI(ABI), ABIName(model::ABI::getName(ABI.ABI())) {}

  std::optional<StateMap> run(const StateMap &InputState) {
    if (ABI.ArgumentsArePositionBased())
      return runForPositionBasedABIs(InputState);
    else
      return runForNonPositionBasedABIs(InputState);
  }

private:
  std::optional<StateMap> runForPositionBasedABIs(const StateMap &State) {
    // Copy the state to serve as a return value.
    StateMap Result = State;

    if (!runForPositionBasedArguments(Result))
      return std::nullopt;

    if (!runForPositionBasedReturnValues(Result))
      return std::nullopt;

    // Check whether the input state is valid and set all the registers still
    // remaining as "unknown" to `No`.
    for (auto [Register, RegisterState] : Result) {
      if (auto &A = accessArgument(RegisterState))
        revng_assert(A != abi::RegisterState::Invalid);
      else
        A = abi::RegisterState::No;

      if (auto &RV = accessReturnValue(RegisterState))
        revng_assert(RV != abi::RegisterState::Invalid);
      else
        RV = abi::RegisterState::No;
    }

    if (!checkPositionBasedABIConformance(Result))
      return std::nullopt;

    return Result;
  }

  bool checkPositionBasedABIConformance(StateMap &State) {
    llvm::ArrayRef GPAR = ABI.GeneralPurposeArgumentRegisters();
    llvm::ArrayRef VAR = ABI.VectorArgumentRegisters();
    llvm::ArrayRef GPRV = ABI.GeneralPurposeReturnValueRegisters();
    llvm::ArrayRef VRVR = ABI.VectorReturnValueRegisters();

    auto AllowedArgumentRegisters = llvm::concat<CRegister>(GPAR, VAR);
    auto AllowedReturnValueRegisters = llvm::concat<CRegister>(GPRV, VRVR);
    for (auto [Register, RegisterState] : State) {
      auto &[UsedForPassingArguments, UsedForReturningValues] = RegisterState;

      revng_assert(UsedForPassingArguments != abi::RegisterState::Invalid);
      if (abi::RegisterState::isYesOrDead(UsedForPassingArguments)) {
        if (!llvm::is_contained(AllowedArgumentRegisters, Register)) {
          if constexpr (EnforceABIConformance == true) {
            revng_log(Log,
                      "Enforcing `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` to `No` as `" << ABIName
                        << "` ABI doesn't allow it to be used.");
            UsedForPassingArguments = abi::RegisterState::No;
          } else {
            revng_log(Log,
                      "Aborting, `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` register is used despite not being allowed by `"
                        << ABIName << "` ABI.");
            return false;
          }
        }
      }

      revng_assert(UsedForReturningValues != abi::RegisterState::Invalid);
      if (abi::RegisterState::isYesOrDead(UsedForReturningValues)) {
        if (!llvm::is_contained(AllowedReturnValueRegisters, Register)) {
          if constexpr (EnforceABIConformance == true) {
            revng_log(Log,
                      "Enforcing `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` to `No` as `" << ABIName
                        << "` ABI doesn't allow it to be used.");
            UsedForReturningValues = abi::RegisterState::No;
          } else {
            revng_log(Log,
                      "Aborting, `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` register is used despite not being allowed by `"
                        << ABIName << "` ABI.");
            return false;
          }
        }
      }
    }

    return true;
  }

  template<AccessorType Accessor>
  void offsetPositionBasedArgument(model::Register::Values Register,
                                   bool &IsRequired,
                                   StateMap &State) {
    auto &Accessed = Accessor(State[Register]);
    if (Accessed == abi::RegisterState::Invalid)
      Accessed = abi::RegisterState::Maybe;

    if (IsRequired == true) {
      if (abi::RegisterState::isYesOrDead(Accessed)) {
        // Nothing to do, it's already set.
      } else {
        // Set it.
        Accessed = abi::RegisterState::YesOrDead;
      }
    } else {
      if (abi::RegisterState::isYesOrDead(Accessed)) {
        // Make all the subsequent registers required.
        IsRequired = true;
      } else {
        // Nothing to do, it doesn't have to be set.
      }
    }
  }

  bool runForPositionBasedArguments(StateMap &State) {
    llvm::ArrayRef GPAR = ABI.GeneralPurposeArgumentRegisters();
    llvm::ArrayRef VAR = ABI.VectorArgumentRegisters();

    bool IsRequired = false;
    if (GPAR.size() > VAR.size()) {
      for (auto Regist : llvm::reverse(GPAR.drop_front(VAR.size())))
        offsetPositionBasedArgument<accessArgument>(Regist, IsRequired, State);
      GPAR = GPAR.take_front(VAR.size());
    } else if (VAR.size() > GPAR.size()) {
      for (auto Regist : llvm::reverse(VAR.drop_front(GPAR.size())))
        offsetPositionBasedArgument<accessArgument>(Regist, IsRequired, State);
      VAR = VAR.take_front(GPAR.size());
    }

    auto ArgumentRange = llvm::zip(llvm::reverse(GPAR), llvm::reverse(VAR));
    for (auto [GPRegister, VRegister] : ArgumentRange) {
      if (auto D = singlePositionBasedDeduction<accessCArgument>(GPRegister,
                                                                 VRegister,
                                                                 IsRequired,
                                                                 State)) {
        accessArgument(State[GPRegister]) = D->GPR;
        accessArgument(State[VRegister]) = D->VR;
      } else {
        return false;
      }
    }

    return true;
  }

  bool runForPositionBasedReturnValues(StateMap &State) {
    llvm::ArrayRef GPRV = ABI.GeneralPurposeReturnValueRegisters();
    llvm::ArrayRef VRVR = ABI.VectorReturnValueRegisters();

    size_t GPRVCount = findLastUsedIndex<accessCReturnValue>(GPRV, State);
    auto UsedGPRV = llvm::ArrayRef(GPRV).take_front(GPRVCount);
    size_t VRVRCount = findLastUsedIndex<accessCReturnValue>(VRVR, State);
    auto UsedVRVR = llvm::ArrayRef(VRVR).take_front(VRVRCount);

    auto AllowedRetValRegisters = llvm::concat<CRegister>(GPRV, VRVR);
    auto UsedRetValRegisters = llvm::concat<CRegister>(UsedGPRV, UsedVRVR);

    // Even for position based ABIs, return values have behaviour patterns
    // similar to non-position based ones, so the initial deduction step is
    // to run the non-position based deduction.
    for (auto [Register, RegisterState] : State) {
      auto &RVState = accessReturnValue(RegisterState);
      bool IsRVAllowed = llvm::is_contained(AllowedRetValRegisters, Register);
      bool IsRVRequired = llvm::is_contained(UsedRetValRegisters, Register);
      auto Deduced = singleNonPositionBasedDeduction(RVState,
                                                     IsRVAllowed,
                                                     IsRVRequired,
                                                     Register);
      if (Deduced == abi::RegisterState::Invalid)
        return false;
      else
        RVState = Deduced;
    }

    // Then we make sure that only either GPRs or VRs are used, never both.
    bool Dummy = false;
    if (auto D = singlePositionBasedDeduction<accessCReturnValue>(GPRV.front(),
                                                                  VRVR.front(),
                                                                  Dummy,
                                                                  State)) {
      if (abi::RegisterState::isYesOrDead(D->GPR)) {
        if (abi::RegisterState::isYesOrDead(D->VR)) {
          revng_log(Log,
                    "Impossible to differentiate whether the return value is "
                    "passed in a general purpose register or in vector ones. "
                    "The ABI is "
                      << ABIName << ".");
          return false;
        } else {
          // The return value is in GPRs, mark all the vector register as `No`.
          for (auto Register : VRVR)
            accessReturnValue(State[Register]) = abi::RegisterState::No;
        }
      } else {
        if (abi::RegisterState::isYesOrDead(D->VR)) {
          // The return value is in vector registers, mark all the GPRs as `No`.
          for (auto Register : GPRV)
            accessReturnValue(State[Register]) = abi::RegisterState::No;
        } else {
          // No return value (???)
          // Since there's no way to confirm, do nothing.
        }
      }
    } else {
      return false;
    }

    return true;
  }

  struct PositionBasedDeductionResult {
    abi::RegisterState::Values GPR;
    abi::RegisterState::Values VR;
  };
  template<CAccessorType Accessor>
  std::optional<PositionBasedDeductionResult>
  singlePositionBasedDeduction(model::Register::Values GPRegister,
                               model::Register::Values VRegister,
                               bool &IsRequired,
                               const StateMap &State) {
    PositionBasedDeductionResult Result;

    Result.GPR = Accessor(State[GPRegister]);
    bool IsGPInvalid = Result.GPR == abi::RegisterState::Invalid ?
                         Result.GPR = abi::RegisterState::Maybe :
                         false;

    Result.VR = Accessor(State[VRegister]);
    bool IsVRInvalid = Result.VR == abi::RegisterState::Invalid ?
                         Result.VR = abi::RegisterState::Maybe :
                         false;

    // If only one register of the pair is specified, consider it to be
    // the dominant one.
    if (IsGPInvalid != IsVRInvalid) {
      if (IsVRInvalid) {
        if (IsRequired) {
          if (!abi::RegisterState::isYesOrDead(Result.GPR))
            Result.GPR = abi::RegisterState::YesOrDead;
          Result.VR = abi::RegisterState::No;
        }
      } else if (IsGPInvalid) {
        if (IsRequired) {
          if (!abi::RegisterState::isYesOrDead(Result.VR))
            Result.VR = abi::RegisterState::YesOrDead;
          Result.GPR = abi::RegisterState::No;
        }
      }
    }

    // Do the deduction
    if (abi::RegisterState::isYesOrDead(Result.GPR)) {
      if (abi::RegisterState::isYesOrDead(Result.VR)) {
        // Both are set - there's no way to tell which one is actually used.
        if constexpr (EnforceABIConformance == true) {
          // Pick one arbitrarily because most likely both of them are `Dead`.
          Result.GPR = abi::RegisterState::YesOrDead;
          Result.VR = abi::RegisterState::No;
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
          return std::nullopt;
        }
      } else {
        // Only GPR is set, ensure VR is marked as `No`.
        Result.VR = abi::RegisterState::No;
        IsRequired = true;
      }
    } else {
      if (abi::RegisterState::isYesOrDead(Result.VR)) {
        // Only VR is set, ensure GPR is marked as `No`.
        Result.GPR = abi::RegisterState::No;
        IsRequired = true;
      } else {
        // Neither one is used.
        if (IsRequired) {
          if constexpr (EnforceABIConformance == true) {
            // Pick one arbitrarily because most likely both of them are `Dead`.
            Result.GPR = abi::RegisterState::YesOrDead;
            Result.VR = abi::RegisterState::No;
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
            return std::nullopt;
          }
        }
      }
    }

    return Result;
  }

  std::optional<StateMap> runForNonPositionBasedABIs(const StateMap &State) {
    if (ABI.OnlyStartDoubleArgumentsFromAnEvenRegister()) {
      // There's a possibility for more in-depth deductions taking the register
      // alignment into consideration.
      // TODO: Investigate this further.
    }

    // Separate all the registers before the "last used one" into separate
    // sub-ranges.

    auto &GPAR = ABI.GeneralPurposeArgumentRegisters();
    size_t GPARCount = findLastUsedIndex<accessCArgument>(GPAR, State);
    auto UsedGPAR = llvm::ArrayRef(GPAR).take_front(GPARCount);

    auto &VAR = ABI.VectorArgumentRegisters();
    size_t VARCount = findLastUsedIndex<accessCArgument>(VAR, State);
    auto UsedVAR = llvm::ArrayRef(VAR).take_front(VARCount);

    auto &GPRVR = ABI.GeneralPurposeReturnValueRegisters();
    size_t GPRVRCount = findLastUsedIndex<accessCReturnValue>(GPRVR, State);
    auto UsedGPRVR = llvm::ArrayRef(GPRVR).take_front(GPRVRCount);

    auto &VRVR = ABI.VectorReturnValueRegisters();
    size_t VRVRCount = findLastUsedIndex<accessCReturnValue>(VRVR, State);
    auto UsedVRVR = llvm::ArrayRef(VRVR).take_front(VRVRCount);

    // Merge sub-ranges together to get the final register sets.
    auto AllowedArgRegisters = llvm::concat<CRegister>(GPAR, VAR);
    auto AllowedRetValRegisters = llvm::concat<CRegister>(GPRVR, VRVR);
    auto UsedArgRegisters = llvm::concat<CRegister>(UsedGPAR, UsedVAR);
    auto UsedRetValRegisters = llvm::concat<CRegister>(UsedGPRVR, UsedVRVR);

    // Copy the state to serve as a return value.
    StateMap Result = State;

    // Try and apply deductions for each register. Abort if any of them fails.
    for (auto [Register, RegisterState] : Result) {
      auto &Argument = accessArgument(RegisterState);
      bool IsArgAllowed = llvm::is_contained(AllowedArgRegisters, Register);
      bool IsArgRequired = llvm::is_contained(UsedArgRegisters, Register);
      auto DeducedArgument = singleNonPositionBasedDeduction(Argument,
                                                             IsArgAllowed,
                                                             IsArgRequired,
                                                             Register);
      if (DeducedArgument == abi::RegisterState::Invalid)
        return std::nullopt;
      else
        Argument = DeducedArgument;

      auto &ReturnValue = accessReturnValue(RegisterState);
      bool IsRVAllowed = llvm::is_contained(AllowedRetValRegisters, Register);
      bool IsRVRequired = llvm::is_contained(UsedRetValRegisters, Register);
      auto DeducedReturnValue = singleNonPositionBasedDeduction(ReturnValue,
                                                                IsRVAllowed,
                                                                IsRVRequired,
                                                                Register);
      if (DeducedReturnValue == abi::RegisterState::Invalid)
        return std::nullopt;
      else
        ReturnValue = DeducedReturnValue;
    }

    return Result;
  }

  abi::RegisterState::Values
  singleNonPositionBasedDeduction(const abi::RegisterState::Values &Input,
                                  bool IsAllowed,
                                  bool IsRequired,
                                  model::Register::Values Register) {
    auto Result = Input; // Copy the input to serve as a return value.

    // Normalize the state.
    if (Result == abi::RegisterState::Invalid)
      Result = abi::RegisterState::Maybe;

    // Fail if the register is not allowed.
    if (!IsAllowed) {
      if constexpr (EnforceABIConformance == true) {
        revng_log(Log,
                  "Enforcing `model::Register::"
                    << model::Register::getName(Register).data()
                    << "` to `No` as `" << ABIName
                    << "` ABI doesn't allow it to be used.");
        return abi::RegisterState::No;
      } else {
        using namespace abi::RegisterState;
        if (Result != Maybe && Result != No) {
          revng_log(Log,
                    "Aborting, `model::Register::"
                      << model::Register::getName(Register).data()
                      << "` register is used despite not being allowed by `"
                      << ABIName << "` ABI.");
          return abi::RegisterState::Invalid;
        } else {
          return abi::RegisterState::No;
        }
      }
    }

    if (abi::RegisterState::isYesOrDead(Result)) {
      // If a usable register is set, it's required by definition.
      revng_assert(IsRequired,
                   "Encountered an impossible state: the data structure is "
                   "probably corrupted.");

      // The current state looks good, preserve it.
      return Result;
    } else {
      if (IsRequired) {
        if (Result == abi::RegisterState::NoOrDead) {
          // The register is required so it must not be marked as `No`.
          // As such `Dead` is the only option left.
          return abi::RegisterState::Dead;
        }

        if (Result == abi::RegisterState::Maybe) {
          // There's no information about the register.
          // Return the most generic acceptable option.
          return abi::RegisterState::YesOrDead;
        }

        if constexpr (EnforceABIConformance == true) {
          // The states are incompatible.
          // Overwrite the result with the most generic acceptable option.
          return abi::RegisterState::YesOrDead;
        } else {
          // Abort if the required register is marked as contradiction.
          if (Result == abi::RegisterState::Contradiction) {
            revng_log(Log,
                      "Aborting, `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` is set to `Contradiction`.");
            return abi::RegisterState::Invalid;
          }

          // Abort if the required register is marked as `No`.
          if (Result == abi::RegisterState::No) {
            revng_log(Log,
                      "Aborting, `model::Register::"
                        << model::Register::getName(Register).data()
                        << "` is set to `No` despite being required.");
            return abi::RegisterState::Invalid;
          }
        }
      } else {
        // The current state looks good, preserve it.
        return Result;
      }
    }

    // Abort in case of unusual circumstances, like a new state enumerator.
    revng_assert("This point should never be reached.");
    return abi::RegisterState::Invalid;
  }
};

using Def = abi::Definition;
std::optional<abi::RegisterState::Map>
Def::tryDeducingRegisterState(const abi::RegisterState::Map &State) const {
  return DeductionImpl<false>(*this).run(State);
}

abi::RegisterState::Map
Def::enforceRegisterState(const abi::RegisterState::Map &State) const {
  auto Result = DeductionImpl<true>(*this).run(State);
  revng_assert(Result != std::nullopt);
  return Result.value();
}
