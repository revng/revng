//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int16_t = !clift.primitive<signed 2>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: is not representable in the target implementation
    clift.expr {
      %0 = clift.imm 0 : !int16_t
      clift.yield %0 : !int16_t
    }
  }
}
