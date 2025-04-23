#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Kind.h"

class pipeline::description::Kind
  : public pipeline::description::generated::Kind {
public:
  using generated::Kind::Kind;
};

#include "revng/Pipeline/Description/Generated/Late/Kind.h"
