#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/SortedVector.h"

/* TUPLE-TREE-YAML
name: CanonicalRegisterValue
doc: |-
  A [`Segment`](#Segment) can specify a set of *canonical values* for certain
  registers.
  This can be used to represent concepts such as the global pointer, which in
  certain ABIs, is not set by each function, but it's possible to assume it has
  a certain value at the entry of the function.
type: struct
fields:
  - name: Register
    doc: The register for which to specify the canonical value.
    type: Register
  - name: Value
    doc: |-
      The canonical value that `Register` can be assumed to hold upon
      function entry.
    type: uint64_t
key:
  - Register
TUPLE-TREE-YAML */

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
