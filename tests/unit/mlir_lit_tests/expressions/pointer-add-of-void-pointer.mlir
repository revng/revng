//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!void$ptr = !clift.pointer<pointer_size = 8, pointee_type = !void>
!ptrdiff_t = !clift.primitive<signed 8>

%p = clift.undef : !void$ptr
%i = clift.imm 0 : !ptrdiff_t

// CHECK: operand pointee must have object type
clift.ptr_add %p, %i : (!void$ptr, !ptrdiff_t)
