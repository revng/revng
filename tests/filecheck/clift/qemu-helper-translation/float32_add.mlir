//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --translate-qemu-helpers | FileCheck %s

!void = !clift.void
!generic32_t = !clift.int<generic 4>
!generic64_t = !clift.int<generic 8>

!float32_add = !clift.func<
  "/helper-function/float32_add" as "float32_add" : !generic32_t(!generic32_t, !generic32_t, !generic64_t)
>

!f = !clift.func<"" : !void(!generic32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  clift.func @float32_add<!float32_add>(!generic32_t, !generic32_t, !generic64_t)

  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !generic32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %0 = clift.use @float32_add : !float32_add
      %0 = clift.use @float32_add : !float32_add
      %1 = clift.undef : !generic64_t
      // CHECK: %1 = clift.bitcast %arg0 : !generic32_t -> !float32_t
      // CHECK: %2 = clift.bitcast %arg0 : !generic32_t -> !float32_t
      // CHECK: %3 = clift.call %0(%1, %2) : !float32_add
      %2 = clift.call %0(%arg0, %arg0, %1) : !float32_add
      // CHECK: %4 = clift.bitcast %3 : !float32_t -> !generic32_t
      // CHECK: clift.yield %4 : !generic32_t
      clift.yield %2 : !generic32_t
    // CHECK: }
    }
  }
// CHECK: }
}
