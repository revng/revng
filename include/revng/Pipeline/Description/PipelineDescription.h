#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/PipelineDescription.h"

class pipeline::description::PipelineDescription
  : public pipeline::description::generated::PipelineDescription {
public:
  using generated::PipelineDescription::PipelineDescription;
};

#include "revng/Pipeline/Description/Generated/Late/PipelineDescription.h"
