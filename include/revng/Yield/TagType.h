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
  - name: Whitespace

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/TagType.h"
#include "revng/Yield/Generated/Late/TagType.h"
