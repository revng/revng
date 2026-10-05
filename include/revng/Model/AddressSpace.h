#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"

#include "revng/Model/Generated/Early/AddressSpace.h"

class model::AddressSpace : public model::generated::AddressSpace {
public:
  using model::generated::AddressSpace::AddressSpace;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/AddressSpace.h"
