//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" as "f" : !void()
>

// CHECK: parameter type must be an object type
!g = !clift.func<
  "/type-definition/1002-CABIFunctionDefinition" as "g" : !void(!f)
>
