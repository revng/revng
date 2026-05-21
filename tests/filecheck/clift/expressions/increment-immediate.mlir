//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %root/bin/revng clift-opt %s 2>&1 | FileCheck %s

!int32_t = !clift.int<signed 4>

%rvalue = clift.imm 1 : !int32_t

// CHECK: failed to verify that operand value is an lvalue expression
clift.inc %rvalue : !int32_t
