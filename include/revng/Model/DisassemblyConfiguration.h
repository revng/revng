#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfigurationAddressStyle.h"

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
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/DisassemblyConfiguration.h"

class model::DisassemblyConfiguration
  : public model::generated::DisassemblyConfiguration {
public:
  using generated::DisassemblyConfiguration::DisassemblyConfiguration;
};

#include "revng/Model/Generated/Late/DisassemblyConfiguration.h"
