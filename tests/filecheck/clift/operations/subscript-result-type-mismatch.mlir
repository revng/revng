//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// A result type unrelated to the pointee is rejected. The generic form is
// needed to write it down at all: the custom one infers the result type from
// the pointer operand, so it can never disagree.

!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>
!p = !clift.ptr<8 to !int32_t>

%i = clift.undef : !int32_t
%0 = clift.undef : !p
// CHECK: 'clift.subscript' op failed to verify that pointer pointee type matches the result type
%1 = "clift.subscript"(%0, %i) : (!p, !int32_t) -> !int64_t
