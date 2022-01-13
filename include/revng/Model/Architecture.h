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

/// Return the size of the pointer in bytes
constexpr inline size_t getPointerSize(Values V) {
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

constexpr inline size_t getCallPushSize(Values V) {
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

} // namespace model::Architecture

#include "revng/Model/Generated/Late/Architecture.h"
