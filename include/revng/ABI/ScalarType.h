#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cstdint>

/* TUPLE-TREE-YAML

name: ScalarType
type: struct
doc: |
  Represents type specific information ABI needs to be aware of,
  for example, alignment.
fields:
  - name: Size
    type: uint64_t

  - name: AlignedAt
    doc: |
      When set to `0` (default), the alignment of this type matches its size
    type: uint64_t
    optional: true

key:
  - Size

TUPLE-TREE-YAML */

#include "revng/ABI/Generated/Early/ScalarType.h"

namespace abi {

class ScalarType : public generated::ScalarType {
public:
  using generated::ScalarType::ScalarType;

  uint64_t alignedAt() const {
    revng_assert(Size() != 0);
    return AlignedAt() != 0 ? AlignedAt() : Size();
  }
};

} // namespace abi

#include "revng/ABI/Generated/Late/ScalarType.h"
