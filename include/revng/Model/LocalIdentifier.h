#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

#include "revng/Model/Generated/Early/LocalIdentifier.h"

class model::LocalIdentifier : public model::generated::LocalIdentifier {
public:
  using generated::LocalIdentifier::LocalIdentifier;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/LocalIdentifier.h"
