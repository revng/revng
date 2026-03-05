//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s

!int32_t = !clift.int<signed 4>
!ptr32_int32_t = !clift.ptr<4 to !int32_t>
!ptr64_int32_t = !clift.ptr<8 to !int32_t>

%value = clift.undef : !ptr32_int32_t

clift.ptr_resize %value : !ptr32_int32_t -> !ptr64_int32_t
