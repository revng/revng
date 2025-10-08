#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"

#include "revng/Model/Generated/Early/BinaryIdentifier.h"

class model::BinaryIdentifier : public model::generated::BinaryIdentifier {
public:
  using model::generated::BinaryIdentifier::BinaryIdentifier;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/BinaryIdentifier.h"
