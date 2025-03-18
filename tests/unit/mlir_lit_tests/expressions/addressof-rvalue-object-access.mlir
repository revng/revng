//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!int32_t = !clift.primitive<signed 4>
!int32_t$ptr = !clift.ptr<8 to !int32_t>

!s = !clift.defined<#clift.struct<
  "/type-definition/1-StructDefinition" : size(4) {
    offset(0) as "x" : !int32_t
  }
>>

%0 = clift.undef : !s
%1 = clift.access<0> %0 : !s -> !int32_t

// CHECK: operand must be an lvalue-expression
%2 = clift.addressof %1 : !int32_t$ptr
