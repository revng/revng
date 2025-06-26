//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!float32_t = !clift.primitive<float 4>
!float32_t$ptr = !clift.ptr<8 to !float32_t>

%f = clift.undef : !float32_t

// CHECK: result must have floating point or integer type
clift.cast<convert> %f : !float32_t -> !float32_t$ptr
