//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!pointer = !clift.pointer<
  pointer_size = 8,
  pointee_type = !clift.primitive<FloatKind 4>>

// CHECK: condition requires an integer type
clift.switch {
  %0 = clift.undef !pointer
  clift.yield %0 : !pointer
}
