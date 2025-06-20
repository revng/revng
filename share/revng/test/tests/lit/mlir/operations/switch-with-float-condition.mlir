//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!float = !clift.primitive<float 4>

// CHECK: condition requires an integer type
clift.switch {
  %0 = clift.undef : !float
  clift.yield %0 : !float
}
