//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>

!g = !clift.func<"/type-definition/1-CABIFunctionDefinition" : !void()>
!ptr32_g = !clift.ptr<4 to !g>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: Pointer operation is not representable in the target implementation
    clift.expr {
      %0 = clift.undef : !ptr32_g
      %1 = clift.call %0() : !ptr32_g
      clift.yield %1 : !void
    }
  }
}
