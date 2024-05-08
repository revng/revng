#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: DisassemblyConfigurationAddressStyle
type: enum
members:
  - name: Smart
    doc: |
      Look for the addresses among basic blocks and functions.
      When a match is found, replace the addresses with relevant labels.
      Otherwise, prints an absolute address instead.

  - name: SmartWithPCRelativeFallback
    doc: |
      Same as \ref Smart, except when unable to single out the target,
      print a PC-relative address instead.

  - name: Strict
    doc: |
      Same as \ref Smart, except when unable to single out the target,
      print an error token.

  - name: Global
    doc: |
      Convert PC relative addresses into global representation.

  - name: PCRelative
    doc: |
      Print all the addresses exactly how disassembler emitted them
      in PC-relative mode.
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/DisassemblyConfigurationAddressStyle.h"
#include "revng/Model/Generated/Late/DisassemblyConfigurationAddressStyle.h"
