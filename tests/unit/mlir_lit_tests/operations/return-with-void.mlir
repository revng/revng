//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!f = !clift.func<
  "/type-definition/1-CABIFunctionDefinition" as "f" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: cannot return expression in function returning void
    clift.return {
      %0 = clift.undef : !void
      clift.yield %0 : !void
    }
  }
}
