//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>

// CHECK: case values must be unique
clift.switch {
  %0 = clift.undef !int32_t
  clift.yield %0 : !int32_t
} case 0 {
} case 0 {
}
