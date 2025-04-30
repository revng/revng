#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Rank.h"

class pipeline::description::Rank
  : public pipeline::description::generated::Rank {
public:
  using generated::Rank::Rank;
};

#include "revng/Pipeline/Description/Generated/Late/Rank.h"
