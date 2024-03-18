#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfigurationAddressStyle.h"
#include "revng/Model/DisassemblyConfigurationImmediateStyle.h"

/* TUPLE-TREE-YAML
name: DisassemblyConfiguration
type: struct
fields:
  - name: DisableEmissionOfInstructionAddress
    type: bool
    optional: true

  - name: DisableEmissionOfRawBytes
    type: bool
    optional: true

  - name: UseATTSyntax
    doc: x86-only.
    type: bool
    optional: true

  - name: AddressStyle
    doc: |
      The default value is `Smart`. See the related enum for further
      explanation.
    type: DisassemblyConfigurationAddressStyle
    optional: true

  - name: ImmediateStyle
    doc: |
      The default value is `CHexadecimal`. See the related enum for further
      explanation.
    type: DisassemblyConfigurationImmediateStyle
    optional: true

  - name: PrintFullMetaAddress
    doc: |
      Set this to true to include the full meta-address whenever one is printed.
      The default value of `false` omits the address type as long as it matches
      that of the binary.
    type: bool
    optional: true

  - name: BasicBlockPrefix
    doc: |
      The prefix attached to the basic block address in the disassembly views.
      The default value is `bb_`.
    type: string
    optional: true

TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/DisassemblyConfiguration.h"

class model::DisassemblyConfiguration
  : public model::generated::DisassemblyConfiguration {
public:
  using generated::DisassemblyConfiguration::DisassemblyConfiguration;
};

#include "revng/Model/Generated/Late/DisassemblyConfiguration.h"
