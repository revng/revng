#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Triple.h"

#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

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

/// Return the size of the pointer in bytes
constexpr inline size_t getPointerSize(Values V) {
  switch (V) {
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

constexpr inline size_t getCallPushSize(Values V) {
  switch (V) {
  case x86:
    return 4;

  case x86_64:
    return 8;

  case arm:
  case mips:
  case aarch64:
  case systemz:
    return 0;

  default:
    revng_abort();
  }
}

} // namespace model::Architecture

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::Architecture::Values>
  : public NamedEnumScalarTraits<model::Architecture::Values> {};
} // namespace llvm::yaml
