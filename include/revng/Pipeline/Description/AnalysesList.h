#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/AnalysesList.h"

class pipeline::description::AnalysesList
  : public pipeline::description::generated::AnalysesList {
public:
  using generated::AnalysesList::AnalysesList;
};

#include "revng/Pipeline/Description/Generated/Late/AnalysesList.h"
