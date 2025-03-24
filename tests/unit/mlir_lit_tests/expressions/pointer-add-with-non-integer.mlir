//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>
!float32_t = !clift.primitive<float 4>

%p = clift.undef : !int32_t$ptr
%i = clift.undef : !float32_t

// CHECK: requires an integer operand
clift.ptr_add %p, %i : (!int32_t$ptr, !float32_t)
