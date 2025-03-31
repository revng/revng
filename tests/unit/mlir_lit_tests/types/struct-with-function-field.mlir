//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

// CHECK: field types must be object types
!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(1) {
    offset(0) : !f
  }
>
