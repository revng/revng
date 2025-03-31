//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<
  "/type-definition/1000-CABIFunctionDefinition" as "f" : !void()
>

// CHECK: cannot return expression in function returning void
clift.module {
  clift.func @f<!f>() {
    clift.return {
      %0 = clift.undef : !int32_t
      clift.yield %0 : !int32_t
    }
  }
}
