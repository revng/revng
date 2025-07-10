//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int16_t = !clift.primitive<signed 2>
!int32_t = !clift.primitive<signed 4>

!uint16_t = !clift.primitive<unsigned 2>
!uint32_t = !clift.primitive<unsigned 4>

%i32 = clift.undef : !int32_t
%u32 = clift.undef : !uint32_t

clift.cast<truncate> %i32 : !int32_t -> !int16_t
clift.cast<truncate> %u32 : !uint32_t -> !uint16_t

!ptr32_int32_t = !clift.ptr<4 to !int32_t>
!ptr64_int32_t = !clift.ptr<8 to !int32_t>

%p64_i32 = clift.undef : !ptr64_int32_t
clift.cast<truncate> %p64_i32 : !ptr64_int32_t -> !ptr32_int32_t
