//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr32 = !clift.ptr<4 to !int32_t>
!int32_t$ptr64 = !clift.ptr<8 to !int32_t>

%p = clift.undef : !int32_t$ptr32

// CHECK: result and argument types must be equal in size
clift.cast<reinterpret> %p : !int32_t$ptr32 -> !int32_t$ptr64
