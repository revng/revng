#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: Rank
doc: Data structure representing a Rank
type: struct
fields:
  - name: Name
    doc: Rank's name
    type: string
  - name: Depth
    doc: Rank's depth
    type: uint64_t
  - name: Parent
    doc: The Rank's parent (if present)
    type: string
    optional: true
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Rank.h"

class pipeline::description::Rank
  : public pipeline::description::generated::Rank {
public:
  using generated::Rank::Rank;
};

#include "revng/Pipeline/Description/Generated/Late/Rank.h"
