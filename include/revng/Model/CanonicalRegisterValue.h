#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"

#include "revng/Model/Generated/Early/CanonicalRegisterValue.h"

namespace model {
class VerifyHelper;
}

class model::CanonicalRegisterValue
  : public model::generated::CanonicalRegisterValue {
public:
  using generated::CanonicalRegisterValue::CanonicalRegisterValue;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/CanonicalRegisterValue.h"
