//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// The index has to be an integer. Unlike the result type, it is spelled out by
// the custom form, so this needs no generic syntax.

!int32_t = !clift.int<signed 4>
!p = !clift.ptr<8 to !int32_t>

%0 = clift.undef : !p
%1 = clift.undef : !p
// CHECK: 'clift.subscript' op operand #1 must be integer type
%2 = clift.subscript %0, %1 : (!p, !p)
