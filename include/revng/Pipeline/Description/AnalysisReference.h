#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/AnalysisReference.h"

class pipeline::description::AnalysisReference
  : public pipeline::description::generated::AnalysisReference {
public:
  using generated::AnalysisReference::AnalysisReference;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysisReference.h"
