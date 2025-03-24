//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

%lvalue = clift.local !int32_t "x"
clift.addressof %lvalue : !int32_t$ptr
