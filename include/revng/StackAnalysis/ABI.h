#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Register.h"
#include "revng/Model/Type.h"

namespace abi {

template<model::abi::Values V>
class ABI;

// TODO: make this as much reusable as possible
// TODO: test

template<>
class ABI<model::abi::SystemV_x86_64> {
public:
  using RegisterState = model::RegisterState::Values;
  using RegisterStateMap = std::map<model::Register::Values,
                                    std::pair<RegisterState, RegisterState>>;

private:
  static constexpr std::array<model::Register::Values, 6> ArgumentRegisters = {
    model::Register::rdi_x86_64, model::Register::rsi_x86_64,
    model::Register::rdx_x86_64, model::Register::rcx_x86_64,
    model::Register::r8_x86_64,  model::Register::r9_x86_64
  };

  static constexpr std::array<model::Register::Values, 2>
    ReturnValueRegisters = { model::Register::rax_x86_64,
                             model::Register::rdx_x86_64 };

  static constexpr std::array<model::Register::Values, 6>
    CalleeSavedRegisters = {
      model::Register::rbx_x86_64, model::Register::rbp_x86_64,
      model::Register::r12_x86_64, model::Register::r13_x86_64,
      model::Register::r14_x86_64, model::Register::r15_x86_64
    };

private:
  struct AnalysisResult {
    bool IsValid;
    uint64_t Arguments;
    uint64_t ReturnValues;
  };

  static AnalysisResult
  analyze(model::Binary &TheBinary, const model::RawFunctionType &Explicit);

public:
  static std::optional<model::RawFunctionType>
  toRaw(model::Binary &TheBinary, const model::CABIFunctionType &Original);

  static bool isCompatible(model::Binary &TheBinary,
                           const model::RawFunctionType &Explicit);

  static std::optional<model::CABIFunctionType>
  toCABI(model::Binary &TheBinary, const model::RawFunctionType &Explicit);

  static model::RawFunctionType indirectCallPrototype(model::Binary &TheBinary);

  void applyDeductions(RegisterStateMap &Prototype);
};

} // namespace abi
