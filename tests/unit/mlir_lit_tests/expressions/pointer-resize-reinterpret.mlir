//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>
!uint32_t = !clift.int<unsigned 4>

!ptr32_int32_t = !clift.ptr<4 to !int32_t>
!ptr64_uint32_t = !clift.ptr<8 to !uint32_t>

%value = clift.undef : !ptr32_int32_t

// CHECK: failed to verify that all of {value, result} have same pointee type
clift.ptr_resize %value : !ptr32_int32_t -> !ptr64_uint32_t
