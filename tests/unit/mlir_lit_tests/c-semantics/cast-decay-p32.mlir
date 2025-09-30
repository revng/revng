//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>
!array_int32_t = !clift.array<1 x !int32_t>
!ptr32_int32_t = !clift.ptr<4 to !clift.primitive<signed 4>>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !array_int32_t
    // CHECK: Pointer operation is not representable in the target implementation
    clift.expr {
      %1 = clift.cast<decay> %0 : !array_int32_t -> !ptr32_int32_t
      clift.yield %1 : !ptr32_int32_t
    }
  }
}
