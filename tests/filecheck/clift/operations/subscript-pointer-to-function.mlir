//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// Subscripting needs an object type, and a function is not one. Indirection on
// the same pointer is valid, because `*fp` is a function designator while
// `fp[i]` is not valid C.

!void = !clift.void
!int32_t = !clift.int<signed 4>
!f = !clift.func<"/type-definition/1004-CABIFunctionDefinition" : !void()>
!p_f = !clift.ptr<8 to !f>

%i = clift.undef : !int32_t
%0 = clift.undef : !p_f
// CHECK: 'clift.subscript' op result #0 must be object type
%1 = clift.subscript %0, %i : (!p_f, !int32_t)
