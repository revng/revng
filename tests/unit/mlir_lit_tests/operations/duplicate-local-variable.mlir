//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    clift.local !int32_t "x"
    // CHECK: conflicts with another local variable
    clift.local !int32_t "x"
  }
}
