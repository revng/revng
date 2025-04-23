#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Analysis.h"

class pipeline::description::Analysis
  : public pipeline::description::generated::Analysis {
public:
  using generated::Analysis::Analysis;
};

#include "revng/Pipeline/Description/Generated/Late/Analysis.h"
