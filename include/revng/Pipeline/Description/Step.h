#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Step.h"

class pipeline::description::Step
  : public pipeline::description::generated::Step {
public:
  using generated::Step::Step;
};

#include "revng/Pipeline/Description/Generated/Late/Step.h"
