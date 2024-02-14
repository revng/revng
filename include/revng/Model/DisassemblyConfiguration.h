#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/DisassemblyConfiguration.h"

class model::DisassemblyConfiguration
  : public model::generated::DisassemblyConfiguration {
public:
  using generated::DisassemblyConfiguration::DisassemblyConfiguration;
};

#include "revng/Model/Generated/Late/DisassemblyConfiguration.h"
