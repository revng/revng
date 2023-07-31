#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: AnalysisReference
doc: A reference to an analysis
type: struct
fields:
  - name: Step
    doc: The step where the analysis belongs to
    type: string
  - name: Name
    doc: The name of the analysis
    type: string
key:
  - Step
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/AnalysisReference.h"

class pipeline::description::AnalysisReference
  : public pipeline::description::generated::AnalysisReference {
public:
  using generated::AnalysisReference::AnalysisReference;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisReference.h"
