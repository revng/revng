//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

!void = !clift.void
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!float32_to_float64 = !clift.func<
  "/helper-function/float32_to_float64" as "float32_to_float64" : !float64_t(!float32_t)
>

!float64_to_float32 = !clift.func<
  "/helper-function/float64_to_float32" as "float64_to_float32" : !float32_t(!float64_t)
>

!f = !clift.func<"" : !void(!float32_t, !float64_t)>

module attributes {clift.module} {
  clift.func @float32_to_float64<!float32_to_float64>(!float32_t)
  clift.func @float64_to_float32<!float64_to_float32>(!float64_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%f32 : !float32_t, %f64 : !float64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_float64 : !float32_to_float64
      // CHECK-NEXT: %0 = clift.ftof %arg0 : !float32_t -> !float64_t
      %1 = clift.call %0(%f32) : !float32_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_float32 : !float64_to_float32
      // CHECK-NEXT: %0 = clift.ftof %arg1 : !float64_t -> !float32_t
      %1 = clift.call %0(%f64) : !float64_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }
  }
}
