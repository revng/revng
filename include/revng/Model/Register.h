#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/Model/Architecture.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: Register
type: enum
members:
  # x86 registers
  - name: eax_x86
  - name: ebx_x86
  - name: ecx_x86
  - name: edx_x86
  - name: esi_x86
  - name: edi_x86
  - name: ebp_x86
  - name: esp_x86
  # x86-64 registers
  - name: rax_x86_64
  - name: rbx_x86_64
  - name: rcx_x86_64
  - name: rdx_x86_64
  - name: rbp_x86_64
  - name: rsp_x86_64
  - name: rsi_x86_64
  - name: rdi_x86_64
  - name: r8_x86_64
  - name: r9_x86_64
  - name: r10_x86_64
  - name: r11_x86_64
  - name: r12_x86_64
  - name: r13_x86_64
  - name: r14_x86_64
  - name: r15_x86_64
  - name: xmm0_x86_64
  - name: xmm1_x86_64
  - name: xmm2_x86_64
  - name: xmm3_x86_64
  - name: xmm4_x86_64
  - name: xmm5_x86_64
  - name: xmm6_x86_64
  - name: xmm7_x86_64
  # ARM registers
  - name: r0_arm
  - name: r1_arm
  - name: r2_arm
  - name: r3_arm
  - name: r4_arm
  - name: r5_arm
  - name: r6_arm
  - name: r7_arm
  - name: r8_arm
  - name: r9_arm
  - name: r10_arm
  - name: r11_arm
  - name: r12_arm
  - name: r13_arm
  - name: r14_arm
  # AArch64 registers
  - name: x0_aarch64
  - name: x1_aarch64
  - name: x2_aarch64
  - name: x3_aarch64
  - name: x4_aarch64
  - name: x5_aarch64
  - name: x6_aarch64
  - name: x7_aarch64
  - name: x8_aarch64
  - name: x9_aarch64
  - name: x10_aarch64
  - name: x11_aarch64
  - name: x12_aarch64
  - name: x13_aarch64
  - name: x14_aarch64
  - name: x15_aarch64
  - name: x16_aarch64
  - name: x17_aarch64
  - name: x18_aarch64
  - name: x19_aarch64
  - name: x20_aarch64
  - name: x21_aarch64
  - name: x22_aarch64
  - name: x23_aarch64
  - name: x24_aarch64
  - name: x25_aarch64
  - name: x26_aarch64
  - name: x27_aarch64
  - name: x28_aarch64
  - name: x29_aarch64
  - name: lr_aarch64
  - name: sp_aarch64
  # MIPS registers
  - name: v0_mips
  - name: v1_mips
  - name: a0_mips
  - name: a1_mips
  - name: a2_mips
  - name: a3_mips
  - name: s0_mips
  - name: s1_mips
  - name: s2_mips
  - name: s3_mips
  - name: s4_mips
  - name: s5_mips
  - name: s6_mips
  - name: s7_mips
  - name: gp_mips
  - name: sp_mips
  - name: fp_mips
  - name: ra_mips
  # SystemZ registers
  - name: r0_systemz
  - name: r1_systemz
  - name: r2_systemz
  - name: r3_systemz
  - name: r4_systemz
  - name: r5_systemz
  - name: r6_systemz
  - name: r7_systemz
  - name: r8_systemz
  - name: r9_systemz
  - name: r10_systemz
  - name: r11_systemz
  - name: r12_systemz
  - name: r13_systemz
  - name: r14_systemz
  - name: r15_systemz
  - name: f0_systemz
  - name: f1_systemz
  - name: f2_systemz
  - name: f3_systemz
  - name: f4_systemz
  - name: f5_systemz
  - name: f6_systemz
  - name: f7_systemz
  - name: f8_systemz
  - name: f9_systemz
  - name: f10_systemz
  - name: f11_systemz
  - name: f12_systemz
  - name: f13_systemz
  - name: f14_systemz
  - name: f15_systemz
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Register.h"

namespace model::Register {

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

namespace model::Register {
inline Values fromRegisterName(llvm::StringRef Name,
                               model::Architecture::Values Architecture) {
  std::string FullName = (llvm::Twine(Name) + "_"
                          + model::Architecture::getName(Architecture))
                           .str();

  return fromName(FullName);
}

} // namespace model::Register

template<>
inline model::Register::Values
getInvalidValueFromYAMLScalar<model::Register::Values>() {
  return model::Register::Invalid;
}

#include "revng/Model/Generated/Late/Register.h"
