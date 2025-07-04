//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!ptr32_int32_t = !clift.ptr<4 to !int32_t>
!ptr64_int32_t = !clift.ptr<8 to !int32_t>

%value = clift.undef : !ptr32_int32_t

// CHECK: result type must be narrower than the argument type
clift.cast<truncate> %value : !ptr32_int32_t -> !ptr64_int32_t
