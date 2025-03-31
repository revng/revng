//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!s = !clift.struct<
  "/type-definition/1-StructDefinition" : size(8) {
    offset(1) : !clift.ptr<4 to !void>
  }
>

clift.module {
  // CHECK: Pointer type is not representable in the target implementation.
  clift.global !s @x
}
