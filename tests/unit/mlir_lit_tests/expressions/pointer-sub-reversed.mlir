//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>
!ptrdiff_t = !clift.primitive<signed 8>

%p = clift.undef : !int32_t$ptr
%i = clift.imm 0 : !ptrdiff_t

// CHECK: left operand must have pointer type
clift.ptr_sub %i, %p : (!ptrdiff_t, !int32_t$ptr)
