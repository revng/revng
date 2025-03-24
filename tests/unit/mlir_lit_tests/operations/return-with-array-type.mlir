//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!array = !clift.array<element_type = !int32_t, elements_count = 1>

// CHECK: requires void or non-array object type
clift.return {
  %0 = clift.undef : !array
  clift.yield %0 : !array
}
