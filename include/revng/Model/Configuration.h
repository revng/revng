#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfiguration.h"

/* TUPLE-TREE-YAML
name: Configuration
type: struct
fields:
  - name: Disassembly
    type: DisassemblyConfiguration
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Configuration.h"

class model::Configuration : public model::generated::Configuration {
public:
  using generated::Configuration::Configuration;
};

#include "revng/Model/Generated/Late/Configuration.h"
