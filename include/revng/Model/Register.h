#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/Architecture.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/EnumSwitch.h"
#include "revng/Support/Generator.h"
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
  - name: st0_x86
  - name: xmm0_x86
  - name: xmm1_x86
  - name: xmm2_x86
  - name: xmm3_x86
  - name: xmm4_x86
  - name: xmm5_x86
  - name: xmm6_x86
  - name: xmm7_x86
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
  - name: fs_x86_64
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
  - name: r15_arm
  - name: q0_arm
  - name: q1_arm
  - name: q2_arm
  - name: q3_arm
  - name: q4_arm
  - name: q5_arm
  - name: q6_arm
  - name: q7_arm
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
  - name: v0_aarch64
  - name: v1_aarch64
  - name: v2_aarch64
  - name: v3_aarch64
  - name: v4_aarch64
  - name: v5_aarch64
  - name: v6_aarch64
  - name: v7_aarch64
  - name: v8_aarch64
  - name: v9_aarch64
  - name: v10_aarch64
  - name: v11_aarch64
  - name: v12_aarch64
  - name: v13_aarch64
  - name: v14_aarch64
  - name: v15_aarch64
  - name: v16_aarch64
  - name: v17_aarch64
  - name: v18_aarch64
  - name: v19_aarch64
  - name: v20_aarch64
  - name: v21_aarch64
  - name: v22_aarch64
  - name: v23_aarch64
  - name: v24_aarch64
  - name: v25_aarch64
  - name: v26_aarch64
  - name: v27_aarch64
  - name: v28_aarch64
  - name: v29_aarch64
  - name: v30_aarch64
  - name: v31_aarch64
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
  - name: t0_mips
  - name: t1_mips
  - name: t2_mips
  - name: t3_mips
  - name: t4_mips
  - name: t5_mips
  - name: t6_mips
  - name: t7_mips
  - name: t8_mips
  - name: t9_mips
  - name: gp_mips
  - name: sp_mips
  - name: fp_mips
  - name: ra_mips
  - name: f0_mips
  - name: f1_mips
  - name: f2_mips
  - name: f3_mips
  - name: f4_mips
  - name: f5_mips
  - name: f6_mips
  - name: f7_mips
  - name: f8_mips
  - name: f9_mips
  - name: f10_mips
  - name: f11_mips
  - name: f12_mips
  - name: f13_mips
  - name: f14_mips
  - name: f15_mips
  - name: f16_mips
  - name: f17_mips
  - name: f18_mips
  - name: f19_mips
  - name: f20_mips
  - name: f21_mips
  - name: f22_mips
  - name: f23_mips
  - name: f24_mips
  - name: f25_mips
  - name: f26_mips
  - name: f27_mips
  - name: f28_mips
  - name: f29_mips
  - name: f30_mips
  - name: f31_mips
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

constexpr inline model::Architecture::Values
getReferenceArchitecture(Values V) {
  switch (V) {
  case eax_x86:
  case ebx_x86:
  case ecx_x86:
  case edx_x86:
  case esi_x86:
  case edi_x86:
  case ebp_x86:
  case esp_x86:
  case st0_x86:
  case xmm0_x86:
  case xmm1_x86:
  case xmm2_x86:
  case xmm3_x86:
  case xmm4_x86:
  case xmm5_x86:
  case xmm6_x86:
  case xmm7_x86:
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
  case fs_x86_64:
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
  case r15_arm:
  case q0_arm:
  case q1_arm:
  case q2_arm:
  case q3_arm:
  case q4_arm:
  case q5_arm:
  case q6_arm:
  case q7_arm:
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
  case v0_aarch64:
  case v1_aarch64:
  case v2_aarch64:
  case v3_aarch64:
  case v4_aarch64:
  case v5_aarch64:
  case v6_aarch64:
  case v7_aarch64:
  case v8_aarch64:
  case v9_aarch64:
  case v10_aarch64:
  case v11_aarch64:
  case v12_aarch64:
  case v13_aarch64:
  case v14_aarch64:
  case v15_aarch64:
  case v16_aarch64:
  case v17_aarch64:
  case v18_aarch64:
  case v19_aarch64:
  case v20_aarch64:
  case v21_aarch64:
  case v22_aarch64:
  case v23_aarch64:
  case v24_aarch64:
  case v25_aarch64:
  case v26_aarch64:
  case v27_aarch64:
  case v28_aarch64:
  case v29_aarch64:
  case v30_aarch64:
  case v31_aarch64:
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
  case t0_mips:
  case t1_mips:
  case t2_mips:
  case t3_mips:
  case t4_mips:
  case t5_mips:
  case t6_mips:
  case t7_mips:
  case t8_mips:
  case t9_mips:
  case gp_mips:
  case sp_mips:
  case fp_mips:
  case ra_mips:
  case f0_mips:
  case f1_mips:
  case f2_mips:
  case f3_mips:
  case f4_mips:
  case f5_mips:
  case f6_mips:
  case f7_mips:
  case f8_mips:
  case f9_mips:
  case f10_mips:
  case f11_mips:
  case f12_mips:
  case f13_mips:
  case f14_mips:
  case f15_mips:
  case f16_mips:
  case f17_mips:
  case f18_mips:
  case f19_mips:
  case f20_mips:
  case f21_mips:
  case f22_mips:
  case f23_mips:
  case f24_mips:
  case f25_mips:
  case f26_mips:
  case f27_mips:
  case f28_mips:
  case f29_mips:
  case f30_mips:
  case f31_mips:
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

constexpr inline bool
isUsedInArchitecture(model::Register::Values Register,
                     model::Architecture::Values Architecture) {
  auto ReferenceArchitecture = getReferenceArchitecture(Register);
  revng_assert(Architecture != model::Architecture::Invalid);
  if (Architecture == ReferenceArchitecture)
    return true;

  // `mipsel` uses `mips` registers.
  if (Architecture == model::Architecture::mipsel)
    if (ReferenceArchitecture == model::Architecture::mips)
      return true;

  return false;
}

inline llvm::StringRef getRegisterName(Values V) {
  llvm::StringRef FullName = getName(V);
  auto Architecture = getReferenceArchitecture(V);
  revng_assert(Architecture != model::Architecture::Invalid);
  llvm::StringRef Name = model::Architecture::getName(Architecture);
  revng_assert(FullName.substr(FullName.size() - Name.size()) == Name);
  return FullName.substr(0, FullName.size() - Name.size() - 1);
}

/// Return the size of the register in bytes
inline uint64_t getSize(Values V) {
  model::Architecture::Values Architecture = getReferenceArchitecture(V);

  switch (V) {
  case st0_x86:
    return 10;
  default:
    break;
  }

  // TODO: this does not account for vector registers, but it should eventually.
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

template<model::Architecture::Values Architecture>
constexpr model::Register::Values getFirst() {
  if constexpr (Architecture == model::Architecture::x86)
    return model::Register::eax_x86;
  else if constexpr (Architecture == model::Architecture::arm)
    return model::Register::r0_arm;
  else if constexpr (Architecture == model::Architecture::mips)
    return model::Register::v0_mips;
  else if constexpr (Architecture == model::Architecture::mipsel)
    return model::Register::v0_mips;
  else if constexpr (Architecture == model::Architecture::x86_64)
    return model::Register::rax_x86_64;
  else if constexpr (Architecture == model::Architecture::aarch64)
    return model::Register::x0_aarch64;
  else if constexpr (Architecture == model::Architecture::systemz)
    return model::Register::r0_systemz;
  else
    static_assert(value_always_false<Architecture>::value,
                  "Unsupported architecture");
}

template<model::Architecture::Values Architecture>
constexpr model::Register::Values getLast() {
  if constexpr (Architecture == model::Architecture::x86)
    return model::Register::xmm7_x86;
  else if constexpr (Architecture == model::Architecture::arm)
    return model::Register::q7_arm;
  else if constexpr (Architecture == model::Architecture::mips)
    return model::Register::f31_mips;
  else if constexpr (Architecture == model::Architecture::mipsel)
    return model::Register::f31_mips;
  else if constexpr (Architecture == model::Architecture::x86_64)
    return model::Register::fs_x86_64;
  else if constexpr (Architecture == model::Architecture::aarch64)
    return model::Register::v31_aarch64;
  else if constexpr (Architecture == model::Architecture::systemz)
    return model::Register::f15_systemz;
  else
    static_assert(value_always_false<Architecture>::value,
                  "Unsupported architecture");
}

template<model::Architecture::Values Architecture>
constexpr size_t getCount() {
  return getLast<Architecture>() - getFirst<Architecture>() + 1;
}

constexpr model::Register::Values
getFirst(model::Architecture::Values Architecture) {
  return skippingEnumSwitch<1>(Architecture, []<model::Architecture::Values A> {
    return getFirst<A>();
  });
}

constexpr model::Register::Values
getLast(model::Architecture::Values Architecture) {
  return skippingEnumSwitch<1>(Architecture, []<model::Architecture::Values A> {
    return getLast<A>();
  });
}

constexpr size_t getCount(model::Architecture::Values Architecture) {
  return skippingEnumSwitch<1>(Architecture, []<model::Architecture::Values A> {
    return getCount<A>();
  });
}

inline model::Architecture::Values
canonicalArchitecture(model::Architecture::Values Architecture) {
  if (Architecture == model::Architecture::mipsel)
    return model::Architecture::mips;

  return Architecture;
}

inline Values fromRegisterName(llvm::StringRef Name,
                               model::Architecture::Values Architecture) {
  Architecture = canonicalArchitecture(Architecture);
  std::string FullName = Name.str();
  FullName += "_";
  FullName += model::Architecture::getName(Architecture).str();

  return fromName(FullName);
}

inline std::optional<unsigned> getMContextIndex(Values V) {
  switch (V) {
  case rax_x86_64:
    return 0xD;
  case rbx_x86_64:
    return 0xB;
  case rcx_x86_64:
    return 0xE;
  case rdx_x86_64:
    return 0xC;
  case rbp_x86_64:
    return 0xA;
  case rsp_x86_64:
    return 0xF;
  case rsi_x86_64:
    return 0x9;
  case rdi_x86_64:
    return 0x8;
  case r8_x86_64:
    return 0x0;
  case r9_x86_64:
    return 0x1;
  case r10_x86_64:
    return 0x2;
  case r11_x86_64:
    return 0x3;
  case r12_x86_64:
    return 0x4;
  case r13_x86_64:
    return 0x5;
  case r14_x86_64:
    return 0x6;
  case r15_x86_64:
    return 0x7;
  default:
    break;
  }

  if (getReferenceArchitecture(V) == model::Architecture::x86_64)
    return std::nullopt;
  else
    revng_abort("Not supported for this architecture");
}

inline llvm::StringRef getCSVName(Values V) {
  // TODO: handle xmm0_x86
  switch (V) {
  case st0_x86:
    return "state_0x83c0";
  case xmm0_x86_64:
    return "state_0x8558";
  case xmm1_x86_64:
    return "state_0x8598";
  case xmm2_x86_64:
    return "state_0x85d8";
  case xmm3_x86_64:
    return "state_0x8618";
  case xmm4_x86_64:
    return "state_0x8658";
  case xmm5_x86_64:
    return "state_0x8698";
  case xmm6_x86_64:
    return "state_0x86d8";
  case xmm7_x86_64:
    return "state_0x8718";
  default:
    return model::Register::getRegisterName(V);
  }
}

inline Values fromCSVName(llvm::StringRef Name,
                          model::Architecture::Values Architecture) {
  if (Architecture == model::Architecture::x86_64) {
    // TODO: handle xmm0_x86
    if (Name == "state_0x83c0") {
      return st0_x86;
    } else if (Name == "state_0x8558") {
      return xmm0_x86_64;
    } else if (Name == "state_0x8598") {
      return xmm1_x86_64;
    } else if (Name == "state_0x85d8") {
      return xmm2_x86_64;
    } else if (Name == "state_0x8618") {
      return xmm3_x86_64;
    } else if (Name == "state_0x8658") {
      return xmm4_x86_64;
    } else if (Name == "state_0x8698") {
      return xmm5_x86_64;
    } else if (Name == "state_0x86d8") {
      return xmm6_x86_64;
    } else if (Name == "state_0x8718") {
      return xmm7_x86_64;
    }
  }

  return model::Register::fromRegisterName(Name, Architecture);
}

constexpr inline model::PrimitiveKind::Values primitiveKind(Values V) {
  switch (V) {
  case eax_x86:
  case ebx_x86:
  case ecx_x86:
  case edx_x86:
  case esi_x86:
  case edi_x86:
  case ebp_x86:
  case esp_x86:
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
  case r15_arm:
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
  case t0_mips:
  case t1_mips:
  case t2_mips:
  case t3_mips:
  case t4_mips:
  case t5_mips:
  case t6_mips:
  case t7_mips:
  case t8_mips:
  case t9_mips:
  case gp_mips:
  case sp_mips:
  case fp_mips:
  case ra_mips:
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
    return model::PrimitiveKind::PointerOrNumber;

  case st0_x86:
  case xmm0_x86:
  case xmm1_x86:
  case xmm2_x86:
  case xmm3_x86:
  case xmm4_x86:
  case xmm5_x86:
  case xmm6_x86:
  case xmm7_x86:
  case xmm0_x86_64:
  case xmm1_x86_64:
  case xmm2_x86_64:
  case xmm3_x86_64:
  case xmm4_x86_64:
  case xmm5_x86_64:
  case xmm6_x86_64:
  case xmm7_x86_64:
  case q0_arm:
  case q1_arm:
  case q2_arm:
  case q3_arm:
  case q4_arm:
  case q5_arm:
  case q6_arm:
  case q7_arm:
  case v0_aarch64:
  case v1_aarch64:
  case v2_aarch64:
  case v3_aarch64:
  case v4_aarch64:
  case v5_aarch64:
  case v6_aarch64:
  case v7_aarch64:
  case v8_aarch64:
  case v9_aarch64:
  case v10_aarch64:
  case v11_aarch64:
  case v12_aarch64:
  case v13_aarch64:
  case v14_aarch64:
  case v15_aarch64:
  case v16_aarch64:
  case v17_aarch64:
  case v18_aarch64:
  case v19_aarch64:
  case v20_aarch64:
  case v21_aarch64:
  case v22_aarch64:
  case v23_aarch64:
  case v24_aarch64:
  case v25_aarch64:
  case v26_aarch64:
  case v27_aarch64:
  case v28_aarch64:
  case v29_aarch64:
  case v30_aarch64:
  case v31_aarch64:
  case f0_mips:
  case f1_mips:
  case f2_mips:
  case f3_mips:
  case f4_mips:
  case f5_mips:
  case f6_mips:
  case f7_mips:
  case f8_mips:
  case f9_mips:
  case f10_mips:
  case f11_mips:
  case f12_mips:
  case f13_mips:
  case f14_mips:
  case f15_mips:
  case f16_mips:
  case f17_mips:
  case f18_mips:
  case f19_mips:
  case f20_mips:
  case f21_mips:
  case f22_mips:
  case f23_mips:
  case f24_mips:
  case f25_mips:
  case f26_mips:
  case f27_mips:
  case f28_mips:
  case f29_mips:
  case f30_mips:
  case f31_mips:
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
    return model::PrimitiveKind::Float;

  case fs_x86_64:
    return model::PrimitiveKind::Generic;

  case Count:
  case Invalid:
  default:
    revng_abort();
  }
}

} // namespace model::Register

template<>
inline model::Register::Values
getInvalidValueFromYAMLScalar<model::Register::Values>() {
  return model::Register::Invalid;
}

namespace model::Architecture {

constexpr inline model::Register::Values getStackPointer(Values V) {
  using namespace model::Register;

  switch (V) {
  case x86:
    return esp_x86;

  case x86_64:
    return rsp_x86_64;

  case arm:
    return r13_arm;

  case mips:
  case mipsel:
    return sp_mips;

  case aarch64:
    return sp_aarch64;

  case systemz:
    return r15_systemz;

  default:
    revng_abort();
  }
}

constexpr inline model::Register::Values getSyscallNumberRegister(Values V) {
  using namespace model::Register;

  switch (V) {
  case x86:
    return eax_x86;

  case x86_64:
    return rax_x86_64;

  case arm:
    return r7_arm;

  case mips:
  case mipsel:
    return v0_mips;

  case aarch64:
    return x8_aarch64;

  case systemz:
    return r1_systemz;

  default:
    revng_abort();
  }
}

constexpr inline model::Register::Values getReturnAddressRegister(Values V) {
  using namespace model::Register;

  switch (V) {
  case x86:
  case x86_64:
    return model::Register::Invalid;

  case arm:
    return r14_arm;

  case mips:
  case mipsel:
    return ra_mips;

  case aarch64:
    return lr_aarch64;

  case systemz:
    return r14_systemz;

  default:
    revng_abort();
  }
}

inline std::optional<unsigned>
getPCMContextIndex(model::Architecture::Values V) {
  switch (V) {
  case x86_64:
    return 0x10;

  case arm:
    return 0x12;

  default:
    return std::nullopt;
  }
}

inline cppcoro::generator<model::Register::Values> registers(Values V) {
  using namespace model::Register;
  auto Next = [](model::Register::Values &V) {
    V = static_cast<model::Register::Values>(static_cast<unsigned long>(V) + 1);
  };
  for (auto Register = getFirst(V); Register <= getLast(V); Next(Register)) {
    co_yield Register;
  }
}

} // namespace model::Architecture

namespace model::Register {

inline cppcoro::generator<model::Register::Values> allRegisters() {
  using namespace model::Architecture;

  // TODO: we might want to provide a similar iteration utilities for
  //       architectures so that we don't need to manually specify every
  //       one of them here.
  for (model::Register::Values Register : registers(x86))
    co_yield Register;
  for (model::Register::Values Register : registers(x86_64))
    co_yield Register;
  for (model::Register::Values Register : registers(arm))
    co_yield Register;
  for (model::Register::Values Register : registers(aarch64))
    co_yield Register;
  for (model::Register::Values Register : registers(mips))
    co_yield Register;
  for (model::Register::Values Register : registers(mipsel))
    co_yield Register;
  for (model::Register::Values Register : registers(systemz))
    co_yield Register;
}

} // namespace model::Register

#include "revng/Model/Generated/Late/Register.h"
