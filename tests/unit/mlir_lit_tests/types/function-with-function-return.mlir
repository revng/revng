//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.defined<#clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "f" : !void()
>>

// CHECK: return type must be void or a non-array object type
!g = !clift.defined<#clift.func<
  "/type-definition/1002-CABIFunctionDefinition" as "g" : !f()
>>
