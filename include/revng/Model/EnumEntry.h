#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"

#include "revng/Model/Generated/Early/EnumEntry.h"

namespace model {
class VerifyHelper;
}

class model::EnumEntry : public model::generated::EnumEntry {
public:
  using generated::EnumEntry::EnumEntry;

public:
  // The entry should have a non-empty name, the name should not be a valid
  // alias, and there should not be empty aliases.
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/EnumEntry.h"
