#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/BasicAnalyses/MaterializedValue.h"

class MemoryOracle {
public:
  virtual MaterializedValue load(uint64_t LoadAddress, unsigned LoadSize) = 0;
  virtual ~MemoryOracle() = default;
};
