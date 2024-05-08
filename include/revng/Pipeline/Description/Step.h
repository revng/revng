#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: Step
doc: Data structure representing a Step
type: struct
fields:
  - name: Name
    doc: Step's name
    type: string
  - name: Component
    doc: Step's component
    type: string
    optional: true
  - name: Parent
    doc: The Step's parent (if present)
    type: string
    optional: true
  - name: Analyses
    doc: List of Analyses that the Step provides
    sequence:
      type: SortedVector
      elementType: Analysis
  - name: Artifacts
    doc: The artifacts that this step provides
    type: Artifacts
    optional: true
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Step.h"

class pipeline::description::Step
  : public pipeline::description::generated::Step {
public:
  using generated::Step::Step;
};

#include "revng/Pipeline/Description/Generated/Late/Step.h"
