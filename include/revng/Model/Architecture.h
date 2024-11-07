#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Triple.h"

/* TUPLE-TREE-YAML
name: Architecture
type: enum
members:
  - name: x86
  - name: x86_64
  - name: arm
  - name: aarch64
  - name: mips
  - name: mipsel
  - name: systemz
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Architecture.h"

namespace model::Architecture {

inline bool isLittleEndian(Values V) {
  switch (V) {
  case model::Architecture::x86:
  case model::Architecture::x86_64:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mipsel:
    return true;
  case model::Architecture::mips:
  case model::Architecture::systemz:
    return false;
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
    return model::Architecture::mips;
  case llvm::Triple::mipsel:
    return model::Architecture::mipsel;
  case llvm::Triple::systemz:
    return model::Architecture::systemz;
  default:
    return model::Architecture::Invalid;
  }
}

inline llvm::Triple::ArchType toLLVMArchitecture(Values V) {
  switch (V) {
  case model::Architecture::x86:
    return llvm::Triple::x86;
  case model::Architecture::x86_64:
    return llvm::Triple::x86_64;
  case model::Architecture::arm:
    return llvm::Triple::arm;
  case model::Architecture::aarch64:
    return llvm::Triple::aarch64;
  case model::Architecture::mips:
    return llvm::Triple::mips;
  case model::Architecture::mipsel:
    return llvm::Triple::mipsel;
  case model::Architecture::systemz:
    return llvm::Triple::systemz;
  default:
    revng_abort();
  }
}

/// Return the size of the pointer in bytes
constexpr inline uint64_t getPointerSize(Values V) {
  switch (V) {
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return 4;
  case model::Architecture::x86_64:
  case model::Architecture::aarch64:
  case model::Architecture::systemz:
    return 8;
  default:
    revng_abort();
  }
}

constexpr inline uint64_t getCallPushSize(Values V) {
  switch (V) {
  case x86:
    return 4;

  case x86_64:
    return 8;

  case arm:
  case mips:
  case mipsel:
  case aarch64:
  case systemz:
    return 0;

  default:
    revng_abort();
  }
}

constexpr inline llvm::ArrayRef<char> getBasicBlockEndingPattern(Values V) {
  switch (V) {
  case x86:
  case x86_64:
    return "\xcc";

  case arm:
    // bx lr
    return "\x1e\xff\x2f\xe1";

  case mips:
    // jr ra
    return "\x08\x00\xe0\x03";

  case mipsel:
    // jr ra
    return "\x03\xe0\x00\x08";

  case aarch64:
    // ret
    return "\xc0\x03\x5f\xd6";

  case systemz:
    // TODO
    return "";

  default:
    revng_abort();
  }
}

// TODO: this is libtinycode-specific
constexpr inline llvm::StringRef getSyscallHelper(Values V) {
  switch (V) {
  case x86:
    return "helper_raise_interrupt";

  case x86_64:
    return "helper_syscall";

  case arm:
  case aarch64:
    return "helper_exception_with_syndrome";

  case mips:
  case mipsel:
    return "helper_raise_exception";

  case systemz:
    return "helper_exception";

  default:
    revng_abort();
  }
}

// TODO: this is libtinycode-specific
inline llvm::ArrayRef<uint64_t> getNoReturnSyscallNumbers(Values V) {
  switch (V) {
  case x86: {
    static uint64_t NoReturnSyscalls[] = {
      0xfc, // exit_group
      0x01, // exit
      0x0b // execve
    };
    return NoReturnSyscalls;
  }

  case x86_64: {
    static uint64_t NoReturnSyscalls[] = {
      0xe7, // exit_group
      0x3c, // exit
      0x3b // execve
    };
    return NoReturnSyscalls;
  }

  case arm: {
    static uint64_t NoReturnSyscalls[] = {
      0xf8, // exit_group
      0x1, // exit
      0xb // execve
    };
    return NoReturnSyscalls;
  }

  case aarch64: {
    static uint64_t NoReturnSyscalls[] = {
      0x5e, // exit_group
      0x5d, // exit
      0xdd // execve
    };
    return NoReturnSyscalls;
  }

  case mips:
  case mipsel: {
    static uint64_t NoReturnSyscalls[] = {
      0x1096, // exit_group
      0xfa1, // exit
      0xfab // execve
    };
    return NoReturnSyscalls;
  }

  case systemz: {
    static uint64_t NoReturnSyscalls[] = {
      0xf8, // exit_group
      0x1, // exit
      0xb, // execve
    };
    return NoReturnSyscalls;
  }

  default:
    revng_abort();
  }
}

inline bool hasELFRelocationAddend(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
  case model::Architecture::systemz:
    return true;

  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return false;

  default:
    revng_abort();
  }
}

inline llvm::StringRef getReadRegisterAssembly(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return "movq %REGISTER, $0";

  case model::Architecture::systemz:
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return "";

  default:
    revng_abort();
  }
}

inline llvm::StringRef getWriteRegisterAssembly(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return "movq $0, %REGISTER";

  case model::Architecture::systemz:
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return "";

  default:
    revng_abort();
  }
}

inline llvm::StringRef getJumpAssembly(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return "jmpq *$0";

  case model::Architecture::systemz:
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return "";

  default:
    revng_abort();
  }
}

inline llvm::StringRef getPCCSVName(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
  case model::Architecture::systemz:
  case model::Architecture::x86:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return "pc";

  default:
    revng_abort();
  }
}

inline llvm::StringRef getQEMUName(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return "x86_64";
  case model::Architecture::systemz:
    return "s390x";
  case model::Architecture::x86:
    return "i386";
  case model::Architecture::arm:
    return "arm";
  case model::Architecture::aarch64:
    return "aarch64";
  case model::Architecture::mips:
    return "mips";
  case model::Architecture::mipsel:
    return "mipsel";
  default:
    revng_abort();
  }
}

inline unsigned getMinimalFinalStackOffset(Values V) {
  switch (V) {
  case model::Architecture::x86_64:
    return 8;
  case model::Architecture::x86:
    return 4;
  case model::Architecture::systemz:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
    return 0;
  default:
    revng_abort();
  }
}

inline constexpr llvm::StringRef getAssemblyCommentIndicator(Values V) {
  switch (V) {
  case model::Architecture::x86:
  case model::Architecture::x86_64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
  case model::Architecture::systemz:
    return "#";
  case model::Architecture::arm:
    return "@";
  case model::Architecture::aarch64:
    return "//";
  default:
    revng_abort();
  }
}

inline constexpr llvm::StringRef getAssemblyLabelIndicator(Values V) {
  switch (V) {
  case model::Architecture::x86:
  case model::Architecture::x86_64:
  case model::Architecture::arm:
  case model::Architecture::aarch64:
  case model::Architecture::mips:
  case model::Architecture::mipsel:
  case model::Architecture::systemz:
    return ":";
  default:
    revng_abort();
  }
}

} // namespace model::Architecture

#include "revng/Model/Generated/Late/Architecture.h"
