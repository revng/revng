//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

!void = !clift.void
!float32_t = !clift.float<4>

!float32_add = !clift.func<
  "/helper-function/float32_add" as "float32_add" : !float32_t(!float32_t, !float32_t)
>

!f = !clift.func<"" : !void(!float32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  clift.func @float32_add<!float32_add>(!float32_t, !float32_t)
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !float32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_add : !float32_add
      // CHECK: %0 = clift.fadd %arg0, %arg0 : !float32_t
      %1 = clift.call %0(%arg0, %arg0) : !float32_add
      // CHECK: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    // CHECK: }
    }
  }
// CHECK: }
}
