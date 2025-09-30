//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: not %revngcliftopt --verify-c %s 2>&1 | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!s = !clift.struct<"/type-definition/1-StructDefinition" : size(4) {
  "/type-definition/1-StructDefinition/0" : offset(0) !int32_t
}>

!ptr32_s = !clift.ptr<4 to !s>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void()
>

module attributes {clift.module} {
  clift.func @f<!f>() {
    // CHECK: Pointer operation is not representable in the target implementation
    clift.expr {
      %0 = clift.undef : !ptr32_s
      %1 = clift.access<indirect 0> %0 : !ptr32_s -> !int32_t
      clift.yield %1 : !int32_t
    }
  }
}
