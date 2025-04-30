#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/AnalysisOption.h"

class pipeline::description::AnalysisOption
  : public pipeline::description::generated::AnalysisOption {
public:
  using generated::AnalysisOption::AnalysisOption;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisOption.h"
