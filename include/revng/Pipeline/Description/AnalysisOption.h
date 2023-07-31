#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML
name: AnalysisOption
doc: Data structure representing an analysis' container input
type: struct
fields:
  - name: Name
    doc: AnalysisOption' name
    type: string
  - name: Type
    doc: Type of the option
    type: string
key:
  - Name
TUPLE-TREE-YAML */

#include "revng/Pipeline/Description/Generated/Early/AnalysisOption.h"

class pipeline::description::AnalysisOption
  : public pipeline::description::generated::AnalysisOption {
public:
  using generated::AnalysisOption::AnalysisOption;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisOption.h"
