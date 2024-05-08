#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

/* TUPLE-TREE-YAML
name: AnalysesList
doc: Data structure representing a list of analyses
type: struct
fields:
  - name: Name
    doc: AnalysesList's name
    type: string
  - name: Analyses
    doc: AnalysesList's list of analyses
    sequence:
      type: SortedVector
      elementType: AnalysisReference
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/AnalysesList.h"

class pipeline::description::AnalysesList
  : public pipeline::description::generated::AnalysesList {
public:
  using generated::AnalysesList::AnalysesList;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysesList.h"
