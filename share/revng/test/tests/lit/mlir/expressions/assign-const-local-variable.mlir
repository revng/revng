//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>

%rvalue = clift.imm 0 : !int32_t

// CHECK: operand must be an lvalue-expression
clift.assign %rvalue, %rvalue : !int32_t
