#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: Artifacts
doc: Data structure representing a Artifacts
type: struct
fields:
  - name: Kind
    doc: Artifacts's kind
    type: string
    optional: true
  - name: Container
    doc: Artifacts's container
    type: string
    optional: true
  - name: SingleTargetFilename
    doc: The Artifacts's filename to use for a single element
    type: string
    optional: true
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Artifacts.h"

class pipeline::description::Artifacts
  : public pipeline::description::generated::Artifacts {
public:
  using generated::Artifacts::Artifacts;

  bool isValid() {
    return not(this->Container().empty() or this->Kind().empty()
               or this->SingleTargetFilename().empty());
  }
};

#include "revng/Pipeline/Description/Generated/Late/Artifacts.h"
