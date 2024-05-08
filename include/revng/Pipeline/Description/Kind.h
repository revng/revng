#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: Kind
doc: Data structure representing a Kind
type: struct
fields:
  - name: Name
    doc: Kind's name
    type: string
  - name: Rank
    doc: Kind's rank
    type: string
  - name: Parent
    doc: The Kind's parent (if present)
    type: string
    optional: true
  - name: DefinedLocations
    doc: List of locations that the Kind provides
    sequence:
      type: SortedVector
      elementType: string
  - name: PreferredKinds
    doc: >
      These are the kinds that should be looked into in order to find
      definitions to locations that are not present in the current document
    sequence:
      type: SortedVector
      elementType: string
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Kind.h"

class pipeline::description::Kind
  : public pipeline::description::generated::Kind {
public:
  using generated::Kind::Kind;
};

#include "revng/Pipeline/Description/Generated/Late/Kind.h"
