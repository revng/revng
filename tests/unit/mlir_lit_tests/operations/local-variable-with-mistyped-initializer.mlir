//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<SignedKind 4>
!int64_t = !clift.primitive<SignedKind 8>

// CHECK: initializer type must match the variable type
clift.local !int32_t "x" = {
  %0 = clift.undef : !int64_t
  clift.yield %0 : !int64_t
}
