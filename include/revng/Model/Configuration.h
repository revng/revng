#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfiguration.h"
#include "revng/Model/NamingConfiguration.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: Configuration
type: struct
fields:
  - name: Disassembly
    type: DisassemblyConfiguration
    optional: true
  - name: Naming
    type: NamingConfiguration
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Configuration.h"

class model::Configuration : public model::generated::Configuration {
public:
  using generated::Configuration::Configuration;

  bool verify(VerifyHelper &VH) const;
  bool verify(bool Assert) const debug_function;
  bool verify() const debug_function;
};

#include "revng/Model/Generated/Late/Configuration.h"
