#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Register.h"
#include "revng/Model/Type.h"

namespace abi {

using RegisterState = model::RegisterState::Values;
using RegisterStateMap = std::map<model::Register::Values,
                                  std::pair<RegisterState, RegisterState>>;

template<model::abi::Values V>
class ABI {
public:
  static std::optional<model::RawFunctionType>
  toRaw(model::Binary &TheBinary, const model::CABIFunctionType &Original) {
    return {};
  }

  static std::optional<model::CABIFunctionType>
  toCABI(model::Binary &TheBinary, const model::RawFunctionType &Explicit) {
    return {};
  }

  static model::TypePath indirectCallPrototype(model::Binary &TheBinary) {
    revng_abort();
  }

  void applyDeductions(RegisterStateMap &Prototype) { return; }
};

// TODO: make this as much reusable as possible
// TODO: test

template<>
class ABI<model::abi::SystemV_x86_64> {
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

  // WIP: rename
  static model::TypePath indirectCallPrototype(model::Binary &TheBinary);

  void applyDeductions(RegisterStateMap &Prototype);
};

template<typename T, size_t Index = 0>
auto polyswitch(T Value, const auto &F) {
  constexpr T Current = static_cast<T>(Index);
  if constexpr (Index < static_cast<size_t>(T::Count)) {
    if (Current == Value) {
      return F.template operator()<Current>();
    } else {
      return polyswitch<T, Index + 1>(Value, F);
    }
  } else {
    abort();
    return F.template operator()<T::Count>();
  }
}

inline std::optional<model::RawFunctionType>
getRawFunctionType(model::Binary &TheBinary,
                   const model::CABIFunctionType *CABI) {

  auto MaybeRaw = polyswitch(CABI->ABI, [&]<model::abi::Values A>() {
    return ABI<A>::toRaw(TheBinary, *CABI);
  });

  revng_assert(MaybeRaw);
  return *MaybeRaw;
}

inline std::optional<model::RawFunctionType>
getRawFunctionType(model::Binary &TheBinary, const model::Type *T) {
  using namespace llvm;
  if (auto *Raw = dyn_cast<model::RawFunctionType>(T)) {
    return *Raw;
  } else if (auto *CABI = dyn_cast<model::CABIFunctionType>(T)) {
    return getRawFunctionType(TheBinary, CABI);
  } else {
    revng_abort("getRawFunctionType with non-function type");
  }
}

} // namespace abi
