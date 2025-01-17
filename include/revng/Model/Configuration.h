#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfiguration.h"
#include "revng/Model/NamingConfiguration.h"

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

  # Configuration options that don't belong anywhere else find themselves here.
  # As this list grows, they should be split into their own sub-sections.
  - name: CommentLineWidth
    doc: |
      Sets a recommended comment line width to improve their readability.

      The default value is `80`.

      Set to `-1` for unlimited line size.
    type: uint64_t
    optional: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Configuration.h"

namespace model {
class VerifyHelper;
}

class model::Configuration : public model::generated::Configuration {
public:
  using generated::Configuration::Configuration;

  bool verify(VerifyHelper &VH) const;
  bool verify(bool Assert) const debug_function;
  bool verify() const debug_function;

public:
  uint64_t commentLineWidth() const {
    if (CommentLineWidth() == 0)
      return 80;
    else
      return CommentLineWidth();
  }
};

#include "revng/Model/Generated/Late/Configuration.h"
