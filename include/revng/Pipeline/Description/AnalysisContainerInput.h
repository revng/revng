#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: AnalysisContainerInput
doc: Data structure representing an analysis' container input
type: struct
fields:
  - name: Name
    doc: AnalysisContainerInput' name
    type: string
  - name: AcceptableKinds
    doc: Kinds accepted by the analysis
    sequence:
      type: SortedVector
      elementType: string
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/AnalysisContainerInput.h"

class pipeline::description::AnalysisContainerInput
  : public pipeline::description::generated::AnalysisContainerInput {
public:
  using generated::AnalysisContainerInput::AnalysisContainerInput;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisContainerInput.h"
