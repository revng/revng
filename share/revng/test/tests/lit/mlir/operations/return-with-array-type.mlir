//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!array = !clift.array<1 x !int32_t>

// CHECK: return type cannot be an array type
clift.return {
  %0 = clift.undef : !array
  clift.yield %0 : !array
}
