#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <vector>

#include "revng/ADT/SortedVector.h"

#include "Generated/Early/TestClass.h"

/* TUPLE-TREE-YAML
name: TestClass
type: struct
fields:
  - name: RequiredField
    type: uint64_t
  - name: OptionalField
    type: uint64_t
    optional: true
  - name: EnumField
    type: model::TestEnum::Values
  - name: SequenceField
    sequence:
      type: SortedVector
      elementType: uint64_t
  - name: ReferenceField
    reference:
      pointeeType: uint64_t
      rootType: model::TestClass
key:
  - RequiredField
TUPLE-TREE-YAML */

namespace model {
class TestClass : public model::generated::TestClass {
public:
  using model::generated::TestClass::TestClass;
};
} // namespace model

#include "Generated/Late/TestClass.h"
