//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

%i = clift.undef : !int32_t

// CHECK: function argument must have function or pointer-to-function type
"clift.call"(%i) : (!int32_t) -> (!void)
