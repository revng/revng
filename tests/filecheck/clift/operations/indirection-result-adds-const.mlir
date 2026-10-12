//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

// The match is on the whole type, qualifiers included, so a result that adds
// const to a non-const pointee is rejected.

!int32_t = !clift.int<signed 4>
!const_int32_t = !clift.const<!clift.int<signed 4>>
!p = !clift.ptr<8 to !int32_t>

%0 = clift.undef : !p
// CHECK: 'clift.indirection' op failed to verify that pointer pointee type matches the result type
%1 = "clift.indirection"(%0) : (!p) -> !const_int32_t
