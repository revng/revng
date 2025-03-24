//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.pointer<pointer_size = 8, pointee_type = !int32_t>
!ptrdiff_t = !clift.primitive<signed 8>

%p = clift.undef : !int32_t$ptr
%i = clift.imm 0 : !ptrdiff_t

clift.ptr_add %p, %i : (!int32_t$ptr, !ptrdiff_t)
clift.ptr_add %i, %p : (!ptrdiff_t, !int32_t$ptr)
clift.ptr_sub %p, %i : (!int32_t$ptr, !ptrdiff_t)
clift.ptr_diff %p, %p : !int32_t$ptr -> !ptrdiff_t
