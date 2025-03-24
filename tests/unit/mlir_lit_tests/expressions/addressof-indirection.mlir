//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

%0 = clift.undef : !int32_t$ptr
%1 = clift.indirection %0 : !int32_t$ptr
%2 = clift.addressof %1 : !int32_t$ptr
