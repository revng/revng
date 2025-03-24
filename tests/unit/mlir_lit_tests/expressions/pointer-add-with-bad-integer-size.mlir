//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>

%p = clift.undef : !int32_t$ptr
%i = clift.imm 0 : !int32_t

// CHECK: pointer and integer operand sizes must match
clift.ptr_add %p, %i : (!int32_t$ptr, !int32_t)
