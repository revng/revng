#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/Triple.h"

#include "revng/Model/TupleTree.h"
#include "revng/Support/YAMLTraits.h"

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

inline uint8_t getPointerByteSize(Values V) {
  switch (V) {
  case Invalid:
    return 0U;
  case x86:
    return 4U;
  case x86_64:
    return 8U;
  case arm:
    return 4U;
  case aarch64:
    return 8U;
  case mips:
    return 4U;
  case systemz:
    return 8U;
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

namespace llvm::yaml {
template<>
struct ScalarEnumerationTraits<model::Architecture::Values>
  : public NamedEnumScalarTraits<model::Architecture::Values> {};
} // namespace llvm::yaml
