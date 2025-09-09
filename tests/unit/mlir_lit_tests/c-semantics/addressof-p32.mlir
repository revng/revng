//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>
!ptr32_int32_t = !clift.ptr<4 to !clift.primitive<signed 4>>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    %0 = clift.local : !int32_t
    // CHECK: Pointer operation is not representable in the target implementation
    clift.expr {
      %1 = clift.addressof %0 : !ptr32_int32_t
      clift.yield %1 : !ptr32_int32_t
    }
  }
}
