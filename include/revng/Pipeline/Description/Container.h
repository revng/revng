#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Container.h"

class pipeline::description::Container
  : public pipeline::description::generated::Container {
public:
  using generated::Container::Container;
};

#include "revng/Pipeline/Description/Generated/Late/Container.h"
