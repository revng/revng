//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// And the other way around: a result that drops the const of the pointee is
// rejected too.

!int32_t = !clift.int<signed 4>
!const_int32_t = !clift.const<!clift.int<signed 4>>
!p_const = !clift.ptr<8 to !const_int32_t>

%i = clift.undef : !int32_t
%0 = clift.undef : !p_const
// CHECK: 'clift.subscript' op failed to verify that pointer pointee type matches the result type
%1 = "clift.subscript"(%0, %i) : (!p_const, !int32_t) -> !int32_t
