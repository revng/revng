#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Triple.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

// WIP: move to Model/Architeture.h

namespace model::Architecture {

enum Values { Invalid, x86, x86_64, arm, aarch64, mips, systemz, Count };

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case x86:
    return "x86";
  case x86_64:
    return "x86_64";
  case arm:
    return "arm";
  case aarch64:
    return "aarch64";
  case mips:
    return "mips";
  case systemz:
    return "systemz";
  default:
    revng_abort();
  }
}

inline Values fromLLVMArchitecture(llvm::Triple::ArchType A) {
  switch (A) {
  case llvm::Triple::x86:
    return model::Architecture::x86;
  case llvm::Triple::x86_64:
    return model::Architecture::x86_64;
  case llvm::Triple::arm:
    return model::Architecture::arm;
  case llvm::Triple::aarch64:
    return model::Architecture::aarch64;
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
    return model::Architecture::mips;
  case llvm::Triple::systemz:
    return model::Architecture::systemz;
  default:
    return model::Architecture::Invalid;
  }
}

} // namespace model::Architecture

// WIP: factor out
namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::Architecture::Values> {
  template<typename T>
  static void enumeration(T &IO, model::Architecture::Values &V) {
    using namespace model::Architecture;
    for (unsigned I = 0; I < Count; ++I) {
      auto Value = static_cast<Values>(I);
      IO.enumCase(V, getName(Value).data(), Value);
    }
  }
};
} // namespace llvm::yaml

namespace model::Register {
enum Values {
  Invalid,
  // x86 registers
  eax_x86,
  ebx_x86,
  ecx_x86,
  edx_x86,
  esi_x86,
  edi_x86,
  ebp_x86,
  esp_x86,
  // x86-64 registers
  rax_x86_64,
  rbx_x86_64,
  rcx_x86_64,
  rdx_x86_64,
  rbp_x86_64,
  rsp_x86_64,
  rsi_x86_64,
  rdi_x86_64,
  r8_x86_64,
  r9_x86_64,
  r10_x86_64,
  r11_x86_64,
  r12_x86_64,
  r13_x86_64,
  r14_x86_64,
  r15_x86_64,
  xmm0_x86_64,
  xmm1_x86_64,
  xmm2_x86_64,
  xmm3_x86_64,
  xmm4_x86_64,
  xmm5_x86_64,
  xmm6_x86_64,
  xmm7_x86_64,
  // ARM registers
  r0_arm,
  r1_arm,
  r2_arm,
  r3_arm,
  r4_arm,
  r5_arm,
  r6_arm,
  r7_arm,
  r8_arm,
  r9_arm,
  r10_arm,
  r11_arm,
  r12_arm,
  r13_arm,
  r14_arm,
  // AArch64 registers
  x0_aarch64,
  x1_aarch64,
  x2_aarch64,
  x3_aarch64,
  x4_aarch64,
  x5_aarch64,
  x6_aarch64,
  x7_aarch64,
  x8_aarch64,
  x9_aarch64,
  x10_aarch64,
  x11_aarch64,
  x12_aarch64,
  x13_aarch64,
  x14_aarch64,
  x15_aarch64,
  x16_aarch64,
  x17_aarch64,
  x18_aarch64,
  x19_aarch64,
  x20_aarch64,
  x21_aarch64,
  x22_aarch64,
  x23_aarch64,
  x24_aarch64,
  x25_aarch64,
  x26_aarch64,
  x27_aarch64,
  x28_aarch64,
  x29_aarch64,
  lr_aarch64,
  sp_aarch64,
  // MIPS registers
  v0_mips,
  v1_mips,
  a0_mips,
  a1_mips,
  a2_mips,
  a3_mips,
  s0_mips,
  s1_mips,
  s2_mips,
  s3_mips,
  s4_mips,
  s5_mips,
  s6_mips,
  s7_mips,
  gp_mips,
  sp_mips,
  fp_mips,
  ra_mips,
  // SystemZ registers
  r0_systemz,
  r1_systemz,
  r2_systemz,
  r3_systemz,
  r4_systemz,
  r5_systemz,
  r6_systemz,
  r7_systemz,
  r8_systemz,
  r9_systemz,
  r10_systemz,
  r11_systemz,
  r12_systemz,
  r13_systemz,
  r14_systemz,
  r15_systemz,
  f0_systemz,
  f1_systemz,
  f2_systemz,
  f3_systemz,
  f4_systemz,
  f5_systemz,
  f6_systemz,
  f7_systemz,
  f8_systemz,
  f9_systemz,
  f10_systemz,
  f11_systemz,
  f12_systemz,
  f13_systemz,
  f14_systemz,
  f15_systemz,
  Count
};
} // end namespace model::Register

template<>
struct KeyedObjectTraits<model::Register::Values>
  : public IdentityKeyedObjectTraits<model::Register::Values> {};

namespace model::Register {

inline llvm::StringRef getName(Values V) {
  switch (V) {
  case Invalid:
    return "Invalid";
  case eax_x86:
    return "eax_x86";
  case ebx_x86:
    return "ebx_x86";
  case ecx_x86:
    return "ecx_x86";
  case edx_x86:
    return "edx_x86";
  case esi_x86:
    return "esi_x86";
  case edi_x86:
    return "edi_x86";
  case ebp_x86:
    return "ebp_x86";
  case esp_x86:
    return "esp_x86";
  case rax_x86_64:
    return "rax_x86_64";
  case rbx_x86_64:
    return "rbx_x86_64";
  case rcx_x86_64:
    return "rcx_x86_64";
  case rdx_x86_64:
    return "rdx_x86_64";
  case rbp_x86_64:
    return "rbp_x86_64";
  case rsp_x86_64:
    return "rsp_x86_64";
  case rsi_x86_64:
    return "rsi_x86_64";
  case rdi_x86_64:
    return "rdi_x86_64";
  case r8_x86_64:
    return "r8_x86_64";
  case r9_x86_64:
    return "r9_x86_64";
  case r10_x86_64:
    return "r10_x86_64";
  case r11_x86_64:
    return "r11_x86_64";
  case r12_x86_64:
    return "r12_x86_64";
  case r13_x86_64:
    return "r13_x86_64";
  case r14_x86_64:
    return "r14_x86_64";
  case r15_x86_64:
    return "r15_x86_64";
  case xmm0_x86_64:
    return "xmm0_x86_64";
  case xmm1_x86_64:
    return "xmm1_x86_64";
  case xmm2_x86_64:
    return "xmm2_x86_64";
  case xmm3_x86_64:
    return "xmm3_x86_64";
  case xmm4_x86_64:
    return "xmm4_x86_64";
  case xmm5_x86_64:
    return "xmm5_x86_64";
  case xmm6_x86_64:
    return "xmm6_x86_64";
  case xmm7_x86_64:
    return "xmm7_x86_64";
  case r0_arm:
    return "r0_arm";
  case r1_arm:
    return "r1_arm";
  case r2_arm:
    return "r2_arm";
  case r3_arm:
    return "r3_arm";
  case r4_arm:
    return "r4_arm";
  case r5_arm:
    return "r5_arm";
  case r6_arm:
    return "r6_arm";
  case r7_arm:
    return "r7_arm";
  case r8_arm:
    return "r8_arm";
  case r9_arm:
    return "r9_arm";
  case r10_arm:
    return "r10_arm";
  case r11_arm:
    return "r11_arm";
  case r12_arm:
    return "r12_arm";
  case r13_arm:
    return "r13_arm";
  case r14_arm:
    return "r14_arm";
  case x0_aarch64:
    return "x0_aarch64";
  case x1_aarch64:
    return "x1_aarch64";
  case x2_aarch64:
    return "x2_aarch64";
  case x3_aarch64:
    return "x3_aarch64";
  case x4_aarch64:
    return "x4_aarch64";
  case x5_aarch64:
    return "x5_aarch64";
  case x6_aarch64:
    return "x6_aarch64";
  case x7_aarch64:
    return "x7_aarch64";
  case x8_aarch64:
    return "x8_aarch64";
  case x9_aarch64:
    return "x9_aarch64";
  case x10_aarch64:
    return "x10_aarch64";
  case x11_aarch64:
    return "x11_aarch64";
  case x12_aarch64:
    return "x12_aarch64";
  case x13_aarch64:
    return "x13_aarch64";
  case x14_aarch64:
    return "x14_aarch64";
  case x15_aarch64:
    return "x15_aarch64";
  case x16_aarch64:
    return "x16_aarch64";
  case x17_aarch64:
    return "x17_aarch64";
  case x18_aarch64:
    return "x18_aarch64";
  case x19_aarch64:
    return "x19_aarch64";
  case x20_aarch64:
    return "x20_aarch64";
  case x21_aarch64:
    return "x21_aarch64";
  case x22_aarch64:
    return "x22_aarch64";
  case x23_aarch64:
    return "x23_aarch64";
  case x24_aarch64:
    return "x24_aarch64";
  case x25_aarch64:
    return "x25_aarch64";
  case x26_aarch64:
    return "x26_aarch64";
  case x27_aarch64:
    return "x27_aarch64";
  case x28_aarch64:
    return "x28_aarch64";
  case x29_aarch64:
    return "x29_aarch64";
  case lr_aarch64:
    return "lr_aarch64";
  case sp_aarch64:
    return "sp_aarch64";
  case v0_mips:
    return "v0_mips";
  case v1_mips:
    return "v1_mips";
  case a0_mips:
    return "a0_mips";
  case a1_mips:
    return "a1_mips";
  case a2_mips:
    return "a2_mips";
  case a3_mips:
    return "a3_mips";
  case s0_mips:
    return "s0_mips";
  case s1_mips:
    return "s1_mips";
  case s2_mips:
    return "s2_mips";
  case s3_mips:
    return "s3_mips";
  case s4_mips:
    return "s4_mips";
  case s5_mips:
    return "s5_mips";
  case s6_mips:
    return "s6_mips";
  case s7_mips:
    return "s7_mips";
  case gp_mips:
    return "gp_mips";
  case sp_mips:
    return "sp_mips";
  case fp_mips:
    return "fp_mips";
  case ra_mips:
    return "ra_mips";
  case r0_systemz:
    return "r0_systemz";
  case r1_systemz:
    return "r1_systemz";
  case r2_systemz:
    return "r2_systemz";
  case r3_systemz:
    return "r3_systemz";
  case r4_systemz:
    return "r4_systemz";
  case r5_systemz:
    return "r5_systemz";
  case r6_systemz:
    return "r6_systemz";
  case r7_systemz:
    return "r7_systemz";
  case r8_systemz:
    return "r8_systemz";
  case r9_systemz:
    return "r9_systemz";
  case r10_systemz:
    return "r10_systemz";
  case r11_systemz:
    return "r11_systemz";
  case r12_systemz:
    return "r12_systemz";
  case r13_systemz:
    return "r13_systemz";
  case r14_systemz:
    return "r14_systemz";
  case r15_systemz:
    return "r15_systemz";
  case f0_systemz:
    return "f0_systemz";
  case f1_systemz:
    return "f1_systemz";
  case f2_systemz:
    return "f2_systemz";
  case f3_systemz:
    return "f3_systemz";
  case f4_systemz:
    return "f4_systemz";
  case f5_systemz:
    return "f5_systemz";
  case f6_systemz:
    return "f6_systemz";
  case f7_systemz:
    return "f7_systemz";
  case f8_systemz:
    return "f8_systemz";
  case f9_systemz:
    return "f9_systemz";
  case f10_systemz:
    return "f10_systemz";
  case f11_systemz:
    return "f11_systemz";
  case f12_systemz:
    return "f12_systemz";
  case f13_systemz:
    return "f13_systemz";
  case f14_systemz:
    return "f14_systemz";
  case f15_systemz:
    return "f15_systemz";
  default:
    revng_abort();
  }
}

inline model::Architecture::Values getArchitecture(Values V) {
  switch (V) {
  case eax_x86:
  case ebx_x86:
  case ecx_x86:
  case edx_x86:
  case esi_x86:
  case edi_x86:
  case ebp_x86:
  case esp_x86:
    return model::Architecture::x86;
  case rax_x86_64:
  case rbx_x86_64:
  case rcx_x86_64:
  case rdx_x86_64:
  case rbp_x86_64:
  case rsp_x86_64:
  case rsi_x86_64:
  case rdi_x86_64:
  case r8_x86_64:
  case r9_x86_64:
  case r10_x86_64:
  case r11_x86_64:
  case r12_x86_64:
  case r13_x86_64:
  case r14_x86_64:
  case r15_x86_64:
  case xmm0_x86_64:
  case xmm1_x86_64:
  case xmm2_x86_64:
  case xmm3_x86_64:
  case xmm4_x86_64:
  case xmm5_x86_64:
  case xmm6_x86_64:
  case xmm7_x86_64:
    return model::Architecture::x86_64;
  case r0_arm:
  case r1_arm:
  case r2_arm:
  case r3_arm:
  case r4_arm:
  case r5_arm:
  case r6_arm:
  case r7_arm:
  case r8_arm:
  case r9_arm:
  case r10_arm:
  case r11_arm:
  case r12_arm:
  case r13_arm:
  case r14_arm:
    return model::Architecture::arm;
  case x0_aarch64:
  case x1_aarch64:
  case x2_aarch64:
  case x3_aarch64:
  case x4_aarch64:
  case x5_aarch64:
  case x6_aarch64:
  case x7_aarch64:
  case x8_aarch64:
  case x9_aarch64:
  case x10_aarch64:
  case x11_aarch64:
  case x12_aarch64:
  case x13_aarch64:
  case x14_aarch64:
  case x15_aarch64:
  case x16_aarch64:
  case x17_aarch64:
  case x18_aarch64:
  case x19_aarch64:
  case x20_aarch64:
  case x21_aarch64:
  case x22_aarch64:
  case x23_aarch64:
  case x24_aarch64:
  case x25_aarch64:
  case x26_aarch64:
  case x27_aarch64:
  case x28_aarch64:
  case x29_aarch64:
  case lr_aarch64:
  case sp_aarch64:
    return model::Architecture::aarch64;
  case v0_mips:
  case v1_mips:
  case a0_mips:
  case a1_mips:
  case a2_mips:
  case a3_mips:
  case s0_mips:
  case s1_mips:
  case s2_mips:
  case s3_mips:
  case s4_mips:
  case s5_mips:
  case s6_mips:
  case s7_mips:
  case gp_mips:
  case sp_mips:
  case fp_mips:
  case ra_mips:
    return model::Architecture::mips;
  case r0_systemz:
  case r1_systemz:
  case r2_systemz:
  case r3_systemz:
  case r4_systemz:
  case r5_systemz:
  case r6_systemz:
  case r7_systemz:
  case r8_systemz:
  case r9_systemz:
  case r10_systemz:
  case r11_systemz:
  case r12_systemz:
  case r13_systemz:
  case r14_systemz:
  case r15_systemz:
  case f0_systemz:
  case f1_systemz:
  case f2_systemz:
  case f3_systemz:
  case f4_systemz:
  case f5_systemz:
  case f6_systemz:
  case f7_systemz:
  case f8_systemz:
  case f9_systemz:
  case f10_systemz:
  case f11_systemz:
  case f12_systemz:
  case f13_systemz:
  case f14_systemz:
  case f15_systemz:
    return model::Architecture::systemz;
  case Count:
  case Invalid:
  default:
    revng_abort();
  }
}

inline llvm::StringRef getRegisterName(Values V) {
  llvm::StringRef FullName = getName(V);
  auto Architecture = getArchitecture(V);
  revng_assert(Architecture != model::Architecture::Invalid);
  auto ArchitectureNameSize = model::Architecture::getName(Architecture).size();
  return FullName.substr(0, FullName.size() - ArchitectureNameSize - 1);
}

/// Return the size of the register in bytes
inline size_t getSize(Values V) {
  model::Architecture::Values Architecture = getArchitecture(V);

  switch (Architecture) {
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::mips:
    return 4;
  case model::Architecture::x86_64:
  case model::Architecture::aarch64:
  case model::Architecture::systemz:
    return 8;
  default:
    revng_abort();
  }
}

} // namespace model::Register

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::Register::Values> {
  template<typename T>
  static void enumeration(T &IO, model::Register::Values &V) {
    using namespace model::Register;
    for (unsigned I = 0; I < Count; ++I) {
      auto Value = static_cast<Values>(I);
      IO.enumCase(V, getName(Value).data(), Value);
    }
  }
};
} // namespace llvm::yaml

namespace model::Register {
inline Values fromRegisterName(llvm::StringRef Name,
                               model::Architecture::Values Architecture) {
  std::string FullName = (llvm::Twine(Name) + "_"
                          + model::Architecture::getName(Architecture))
                           .str();

  return getValueFromYAMLScalar<Values>(FullName);
}

} // namespace model::Register

template<>
inline model::Register::Values
getInvalidValueFromYAMLScalar<model::Register::Values>() {
  return model::Register::Invalid;
}

namespace model::RegisterState {

enum Values {
  Invalid,
  No,
  NoOrDead,
  Dead,
  Yes,
  YesOrDead,
  Maybe,
  Contradiction
};

} // namespace model::RegisterState

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::RegisterState::Values> {
  template<typename T>
  static void enumeration(T &IO, model::RegisterState::Values &V) {
    using namespace model::RegisterState;
    IO.enumCase(V, "Invalid", Invalid);
    IO.enumCase(V, "No", No, QuotingType::Double);
    IO.enumCase(V, "NoOrDead", NoOrDead);
    IO.enumCase(V, "Dead", Dead);
    IO.enumCase(V, "Yes", Yes, QuotingType::Double);
    IO.enumCase(V, "YesOrDead", YesOrDead);
    IO.enumCase(V, "Maybe", Maybe);
    IO.enumCase(V, "Contradiction", Contradiction);
  }
};
} // namespace llvm::yaml

namespace model::RegisterState {

inline llvm::StringRef getName(Values V) {
  return getNameFromYAMLEnumScalar(V);
}

inline Values fromName(llvm::StringRef Name) {
  return getValueFromYAMLScalar<Values>(Name);
}

inline bool isYesOrDead(model::RegisterState::Values V) {
  return (V == model::RegisterState::Yes or V == model::RegisterState::YesOrDead
          or V == model::RegisterState::Dead);
}

inline bool shouldEmit(model::RegisterState::Values V) {
  return isYesOrDead(V);
}

} // namespace model::RegisterState
