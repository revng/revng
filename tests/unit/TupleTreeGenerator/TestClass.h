#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
    type: TestEnum
  - name: SequenceField
    sequence:
      type: SortedVector
      elementType: uint64_t
  - name: ReferenceField
    reference:
      pointeeType: uint64_t
      rootType: TestClass
key:
  - RequiredField
TUPLE-TREE-YAML */

namespace ttgtest {
class TestClass : public ttgtest::generated::TestClass {
public:
  using ttgtest::generated::TestClass::TestClass;
};
} // namespace ttgtest

#include "Generated/Late/TestClass.h"
