//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!int32_t$const = !clift.const<!clift.primitive<signed 4>>
!int32_t$const$ptr = !clift.ptr<8 to !int32_t$const>

!ptrdiff_t = !clift.primitive<signed 8>

%p = clift.undef : !int32_t$ptr
%i = clift.imm 0 : !ptrdiff_t

// CHECK: result and pointer operand types must match
"clift.ptr_add"(%p, %i) : (!int32_t$ptr, !ptrdiff_t) -> !int32_t$const$ptr
