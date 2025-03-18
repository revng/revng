//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!s = !clift.defined<#clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {}
>>

// CHECK: condition requires a scalar type
clift.for {} {
  %0 = clift.undef : !s
  clift.yield %0 : !s
} {} {}
