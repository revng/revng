#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML

name: TagType
doc: Enum for identifying different instruction markup tag types
type: enum
members:
  - name: Untagged
  - name: Helper
  - name: Memory
  - name: Register
  - name: Immediate
  - name: Address
  - name: AbsoluteAddress
  - name: PCRelativeAddress
  - name: Mnemonic
  - name: MnemonicPrefix
  - name: MnemonicSuffix
  - name: Directive
  - name: Whitespace
  - name: Label

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/TagType.h"

namespace yield::TagType {

constexpr inline llvm::StringRef toPTML(const Values &V) {
  switch (V) {
  case yield::TagType::Address:
  case yield::TagType::PCRelativeAddress:
  case yield::TagType::AbsoluteAddress:
  case yield::TagType::Immediate:
    return "asm.immediate-value";
  case yield::TagType::Memory:
    return "asm.memory-operand";
  case yield::TagType::Mnemonic:
    return "asm.mnemonic";
  case yield::TagType::MnemonicPrefix:
    return "asm.mnemonic-prefix";
  case yield::TagType::MnemonicSuffix:
    return "asm.mnemonic-suffix";
  case yield::TagType::Register:
    return "asm.register";
  case yield::TagType::Helper:
    return "asm.helper";
  case yield::TagType::Label:
    return "asm.label";
  case yield::TagType::Directive:
    return "asm.directive";
  case yield::TagType::Whitespace:
  case yield::TagType::Untagged:
    return "";
  default:
    revng_abort(("Unknown tag type: " + getName(V).str()).c_str());
  }
}

} // namespace yield::TagType

#include "revng/Yield/Generated/Late/TagType.h"
