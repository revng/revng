#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: PipelineDescription
doc: Data structure representing the structure of the pipeline
type: struct
fields:
  - name: AnalysesLists
    doc: AnalysesLists available in the pipeline
    sequence:
      type: SortedVector
      elementType: AnalysesList
  - name: Globals
    doc: List of Globals
    sequence:
      type: SortedVector
      elementType: string
  - name: Kinds
    doc: Kinds
    sequence:
      type: SortedVector
      elementType: Kind
  - name: Ranks
    doc: Ranks
    sequence:
      type: SortedVector
      elementType: Rank
  - name: Containers
    sequence:
      type: SortedVector
      elementType: Container
  - name: Steps
    sequence:
      type: SortedVector
      elementType: Step
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/PipelineDescription.h"

class pipeline::description::PipelineDescription
  : public pipeline::description::generated::PipelineDescription {
public:
  using generated::PipelineDescription::PipelineDescription;
};

#include "revng/Pipeline/Description/Generated/Late/PipelineDescription.h"
