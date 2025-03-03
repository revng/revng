//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

%0 = clift.imm 42 : !int32_t

// CHECK: operand must be an lvalue-expression
%1 = clift.addressof %0 : !int32_t$ptr
