#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/AnalysisContainerInput.h"

class pipeline::description::AnalysisContainerInput
  : public pipeline::description::generated::AnalysisContainerInput {
public:
  using generated::AnalysisContainerInput::AnalysisContainerInput;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisContainerInput.h"
