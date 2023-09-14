#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: Analysis
doc: Data structure representing an analysis
type: struct
fields:
  - name: Name
    doc: Analysis' name
    type: string
  - name: ContainerInputs
    doc: Analysis' container inputs
    sequence:
      type: SortedVector
      elementType: AnalysisContainerInput
  - name: Options
    doc: Analysis' options
    sequence:
      type: SortedVector
      elementType: AnalysisOption
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/Analysis.h"

class pipeline::description::Analysis
  : public pipeline::description::generated::Analysis {
public:
  using generated::Analysis::Analysis;
};

#include "revng/Pipeline/Description/Generated/Late/Analysis.h"
