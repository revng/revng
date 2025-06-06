//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

%p = clift.undef : !int32_t$ptr

// CHECK: requires exactly one pointer operand
"clift.ptr_add"(%p, %p) : (!int32_t$ptr, !int32_t$ptr) -> !int32_t$ptr
