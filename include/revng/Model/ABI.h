#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Architecture.h"

/* TUPLE-TREE-YAML
name: ABI
type: enum
members:
  - name: SystemV_x86_64
    doc: |
      64-bit SystemV ABI for x86 processor architecture
      ([reference](https://gitlab.com/x86-psABIs/x86-64-ABI)).
  - name: SystemV_x86
    doc: |
      32-bit SystemV ABI for x86 processor architecture ([reference][abi1]).

      [abi1]: https://gitlab.com/x86-psABIs/i386-ABI/-/tree/hjl/x86/master

  - name: SystemV_x86_regparm_3
    doc: |
      A GCC specific modification of the 32-bit SystemV ABI for x86 processor
      architecture. It allows three first GPR-sized arguments to be passed
      using the EAX, EDX, and ECX registers.
      See the reference for `regparm` x86 function attribute.

  - name: SystemV_x86_regparm_2
    doc: |
      A GCC specific modification of the 32-bit SystemV ABI for x86 processor
      architecture. It allows two first GPR-sized arguments to be passed
      using the EAX, and ECX registers.
      See the GCC documentation for `regparm` x86 function attribute.

  - name: SystemV_x86_regparm_1
    doc: |
      A GCC specific modification of the 32-bit SystemV ABI for x86 processor
      architecture. It allows the first GPR-sized argument to be passed
      using the EAX register.
      See the GCC documentation for `regparm` x86 function attribute.

  - name: Microsoft_x86_64
    doc: >
      64-bit Microsoft ABI for x86 processor architecture
      ([reference][abi2]).

      [abi2]: https://docs.microsoft.com/en-us/cpp/build/x64-calling-convention

  - name: Microsoft_x86_64_vectorcall
    doc: |
      A modification of 64-bit Microsoft ABI for x86 processor architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/vectorcall)).
      It allows using extra vector registers for passing function arguments.

  - name: Microsoft_x86_cdecl
    doc: |
      The default 32-bit Microsoft ABI for x86 processor architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/cdecl)).
      It was indented to be compatible with `SystemV_x86` but there are slight
      differences.

  - name: Microsoft_x86_cdecl_gcc
    doc: |
      32-bit Microsoft x86 `cdecl` abi as implemented in GCC (subtly different
      from the original).

  - name: Microsoft_x86_stdcall
    doc: |
      A modification of the 32-bit `__cdecl` Microsoft ABI for x86 processor
      architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/stdcall)).
      The main difference is the fact that the callee is responsible for stack
      cleanup instead of the caller.

  - name: Microsoft_x86_stdcall_gcc
    doc: |
      32-bit Microsoft x86 `stdcall` abi as implemented in GCC (subtly different
      from the original).

  - name: Microsoft_x86_thiscall
    doc: |
      A modification of the 32-bit `__stdcall` Microsoft ABI for x86 processor
      architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/thiscall)).
      The main difference is the fact that it allows to pass a
      single (the first) function argument using a register. This ABI is only
      used for member function call where the first argument is always a `this`
      pointer.

  - name: Microsoft_x86_fastcall
    doc: |
      A modification of the 32-bit `__stdcall` Microsoft ABI for x86 processor
      architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/fastcall)).
      The main difference is the fact that it allows to pass two
      first GPR-sized non-aggregate function arguments in registers.

  - name: Microsoft_x86_fastcall_gcc
    doc: |
      32-bit Microsoft x86 `fastcall` abi as implemented in GCC (subtly
      different from the original).

  - name: Microsoft_x86_vectorcall
    doc: |
      A modification of the 32-bit `__fastcall` Microsoft ABI for x86 processor
      architecture
      ([reference](https://docs.microsoft.com/en-us/cpp/cpp/vectorcall)).
      It allows using extra vector registers for passing function arguments.

  - name: AAPCS64
    doc: |
      Stands for `Arm Architecture Procedure Call Standard (64-bit)`
      ([reference](https://github.com/ARM-software/abi-aa/releases)).
      The official ABI for AArch64 (ARM64) processor architecture.

  - name: Microsoft_AAPCS64
    doc: >
      Stands for "Arm Architecture Procedure Call Standard (64-bit)".
      This represents the version of the ABI used by windows-on-arm.
      For differences from the original ABI see the [reference][ab3].

      [ab3]: https://learn.microsoft.com/cpp/build/arm64-windows-abi-conventions

  - name: Apple_AAPCS64
    doc: "Stands for \"Arm Architecture Procedure Call Standard (64-bit)\".\n
          This represents the version of the ABI used by the apple products.\n
          For differences from the original ABI see the [reference](\
          https://developer.apple.com/documentation/xcode/writing-arm64-
          code-for-apple-platforms)."

  - name: AAPCS
    doc: |
      Stands for "Arm Architecture Procedure Call Standard"
      ([reference](https://github.com/ARM-software/abi-aa/releases)).
      The official ABI for ARM processor architecture.

  - name: SystemV_MIPS_o32
    doc: >
      The ABI for MIPS RISC processor architecture
      ([reference][abi4]).

      [abi4]: http://math-atlas.sourceforge.net/devel/assembly/mipsabi32.pdf

  - name: SystemV_MIPSEL_o32
    doc: >
      The ABI for little-endian edition of the MIPS RISC processor architecture
      ([reference][abi5]).

      [abi5]: http://math-atlas.sourceforge.net/devel/assembly/mipsabi32.pdf

  - name: SystemZ_s390x
    doc: |
      The s390x ABI for SystemZ processor architecture
      ([reference](https://github.com/IBM/s390x-abi)).
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/ABI.h"

namespace model::ABI {

inline constexpr model::Architecture::Values
getArchitecture(model::ABI::Values V) {
  switch (V) {
  case model::ABI::SystemV_x86_64:
  case model::ABI::Microsoft_x86_64:
  case model::ABI::Microsoft_x86_64_vectorcall:
    return model::Architecture::x86_64;

  case model::ABI::SystemV_x86:
  case model::ABI::SystemV_x86_regparm_3:
  case model::ABI::SystemV_x86_regparm_2:
  case model::ABI::SystemV_x86_regparm_1:
  case model::ABI::Microsoft_x86_vectorcall:
  case model::ABI::Microsoft_x86_cdecl:
  case model::ABI::Microsoft_x86_cdecl_gcc:
  case model::ABI::Microsoft_x86_stdcall:
  case model::ABI::Microsoft_x86_stdcall_gcc:
  case model::ABI::Microsoft_x86_fastcall:
  case model::ABI::Microsoft_x86_fastcall_gcc:
  case model::ABI::Microsoft_x86_thiscall:
    return model::Architecture::x86;

  case model::ABI::AAPCS64:
  case model::ABI::Microsoft_AAPCS64:
  case model::ABI::Apple_AAPCS64:
    return model::Architecture::aarch64;
  case model::ABI::AAPCS:
    return model::Architecture::arm;

  case model::ABI::SystemV_MIPS_o32:
    return model::Architecture::mips;

  case model::ABI::SystemV_MIPSEL_o32:
    return model::Architecture::mipsel;

  case model::ABI::SystemZ_s390x:
    return model::Architecture::systemz;

  case model::ABI::Count:
  case model::ABI::Invalid:
  default:
    revng_abort();
  }
}

/// A workaround for the model not having dedicated `mipsel` registers.
///
/// The returned architecture is the one registers of which is used inside
/// the `abi::Definition`.
inline constexpr model::Architecture::Values
getRegisterArchitecture(model::ABI::Values V) {
  if (V == model::ABI::SystemV_MIPSEL_o32)
    return model::Architecture::mips;
  else
    return getArchitecture(V);
}

/// \return the size of the pointer under the specified ABI.
inline constexpr uint64_t getPointerSize(model::ABI::Values V) {
  return model::Architecture::getPointerSize(getArchitecture(V));
}

// TODO: move binary specific information away from a major header
inline constexpr std::optional<model::ABI::Values>
getDefaultForELF(model::Architecture::Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return model::ABI::SystemV_x86_64;
  case model::Architecture::x86:
    return model::ABI::SystemV_x86;
  case model::Architecture::aarch64:
    return model::ABI::AAPCS64;
  case model::Architecture::arm:
    return model::ABI::AAPCS;
  case model::Architecture::mips:
    return model::ABI::SystemV_MIPS_o32;
  case model::Architecture::mipsel:
    return model::ABI::SystemV_MIPSEL_o32;
  case model::Architecture::systemz:
    return model::ABI::SystemZ_s390x;
  default:
    return std::nullopt;
  }
}

inline constexpr std::optional<model::ABI::Values>
getDefaultForPECOFF(model::Architecture::Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return model::ABI::Microsoft_x86_64;
  case model::Architecture::x86:
    return model::ABI::Microsoft_x86_cdecl;
  case model::Architecture::aarch64:
    return model::ABI::Microsoft_AAPCS64;
  default:
    return std::nullopt;
  }
}

inline constexpr std::optional<model::ABI::Values>
getDefaultForMachO(model::Architecture::Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return model::ABI::SystemV_x86_64;
  case model::Architecture::x86:
    return model::ABI::SystemV_x86;
  case model::Architecture::aarch64:
    return model::ABI::Apple_AAPCS64;
  default:
    return std::nullopt;
  }
}

inline constexpr llvm::StringRef getDescription(model::ABI::Values V) {
  switch (V) {
  case model::ABI::SystemV_x86_64:
    return "64-bit SystemV x86 abi";
  case model::ABI::Microsoft_x86_64:
    return "64-bit Microsoft x86 abi";
  case model::ABI::Microsoft_x86_64_vectorcall:
    return "64-bit Microsoft x86 abi with extra vector "
           "registers designited for passing function "
           "arguments";

  case model::ABI::SystemV_x86:
    return "32-bit SystemV x86 abi";
  case model::ABI::SystemV_x86_regparm_3:
    return "32-bit SystemV x86 abi that allows the first three GPR-sized "
           "scalar arguments to be passed using the registers";
  case model::ABI::SystemV_x86_regparm_2:
    return "32-bit SystemV x86 abi that allows the first two GPR-sized "
           "scalar arguments to be passed using the registers";
  case model::ABI::SystemV_x86_regparm_1:
    return "32-bit SystemV x86 abi that allows the first "
           "GPR-sized scalar argument to be passed using "
           "the registers";
  case model::ABI::Microsoft_x86_vectorcall:
    return "64-bit Microsoft x86 abi, it extends `fastcall` "
           "by allowing extra vector registers to be used for "
           "function argument passing";
  case model::ABI::Microsoft_x86_cdecl:
    return "32-bit Microsoft x86 abi that was intended to "
           "mimic 32-bit SystemV x86 abi but has minor "
           "differences";
  case model::ABI::Microsoft_x86_cdecl_gcc:
    return "32-bit Microsoft x86 `cdecl` abi as implemented in GCC (subtly "
           "different from the original).";
  case model::ABI::Microsoft_x86_stdcall:
    return "32-bit Microsoft x86 abi, it is a modification of "
           "`cdecl` that's different in a sense that the "
           "callee is responsible for stack cleanup instead "
           "of the caller";
  case model::ABI::Microsoft_x86_stdcall_gcc:
    return "32-bit Microsoft x86 `stdcall` abi as implemented in GCC (subtly "
           "different from the original).";
  case model::ABI::Microsoft_x86_fastcall:
    return "32-bit Microsoft x86 abi, it extends `stdcall` by "
           "allowing two first GPR-sized function arguments "
           "to be passed using the registers";
  case model::ABI::Microsoft_x86_fastcall_gcc:
    return "32-bit Microsoft x86 `fastcall` abi as implemented in GCC (subtly "
           "different from the original).";
  case model::ABI::Microsoft_x86_thiscall:
    return "32-bit Microsoft x86 abi, it extends `stdcall` by "
           "allowing `this` pointer in method-style calls to "
           "be passed using a register. It is never used for "
           "`free` functions";

  case model::ABI::AAPCS64:
    return "64-bit ARM abi";
  case model::ABI::Microsoft_AAPCS64:
    return "Microsoft version of 64-bit ARM abi";
  case model::ABI::Apple_AAPCS64:
    return "Apple version of 64-bit ARM abi";
  case model::ABI::AAPCS:
    return "32-bit ARM abi";

  case model::ABI::SystemV_MIPS_o32:
    return "The \"old\" 32-bit MIPS abi";

  case model::ABI::SystemV_MIPSEL_o32:
    return "The \"old\" 32-bit MIPS abi (little endian edition)";

  case model::ABI::SystemZ_s390x:
    return "The s390x SystemZ ABI";

  case model::ABI::Count:
  case model::ABI::Invalid:
  default:
    return "Unknown and/or unsupported ABI";
  }
}

} // namespace model::ABI

#include "revng/Model/Generated/Late/ABI.h"
