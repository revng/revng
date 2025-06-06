//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

// CHECK: return type must be an object type or void
clift.return {
  %0 = clift.undef : !f
  clift.yield %0 : !f
}
