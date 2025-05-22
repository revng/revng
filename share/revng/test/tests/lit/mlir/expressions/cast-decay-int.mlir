//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

%int = clift.undef : !int32_t

// CHECK: argument must have array or function type
clift.cast<decay> %int : !int32_t -> !int32_t$ptr
