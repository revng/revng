//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.primitive<signed 4>
!int32_t$array = !clift.array<1 x !int32_t>
!int32_t$ptr = !clift.pointer<pointee_type = !int32_t, pointer_size = 8>

%array = clift.undef : !int32_t$array
clift.cast<decay> %array : !int32_t$array -> !int32_t$ptr
