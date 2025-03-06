//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!float32_t = !clift.primitive<float 4>
!float32_t$ptr = !clift.ptr<8 to !float32_t>

%p = clift.undef : !float32_t$ptr

// CHECK: operand must have floating point or integer type
clift.cast<convert> %p : !float32_t$ptr -> !float32_t
