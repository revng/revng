//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

%v = clift.local : !int32_t
%t = clift.ternary %v, %v, %v : (!int32_t, !int32_t)

// CHECK: failed to verify that operand lhs is an lvalue expression
clift.assign %t, %t : !int32_t
