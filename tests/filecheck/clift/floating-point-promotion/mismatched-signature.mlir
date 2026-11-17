//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

// Calls to helpers whose signature does not match the expected signature are
// not promoted.

!void = !clift.void
!int32_t = !clift.int<signed 4>
!uint8_t = !clift.int<unsigned 1>
!uint32_t = !clift.int<unsigned 4>
!float32_t = !clift.float<4>

// Expects a signed parameter.
!int32_to_float32 = !clift.func<
  "/helper-function/int32_to_float32" as "int32_to_float32" : !float32_t(!uint32_t)
>

// Expects a 64-bit parameter.
!int64_to_float32 = !clift.func<
  "/helper-function/int64_to_float32" as "int64_to_float32" : !float32_t(!int32_t)
>

// Expects 64-bit floating point parameters.
!float64_lt = !clift.func<
  "/helper-function/float64_lt" as "float64_lt" : !uint8_t(!float32_t, !float32_t)
>

!f = !clift.func<"" : !void(!int32_t, !uint32_t, !float32_t)>

module attributes {clift.module} {
  clift.func @int32_to_float32<!int32_to_float32>(!uint32_t)
  clift.func @int64_to_float32<!int64_to_float32>(!int32_t)
  clift.func @float64_lt<!float64_lt>(!float32_t, !float32_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%i32 : !int32_t, %u32 : !uint32_t, %f32 : !float32_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NEXT: %0 = clift.use @int32_to_float32
      %0 = clift.use @int32_to_float32 : !int32_to_float32
      // CHECK-NEXT: %1 = clift.call %0(%arg1)
      %1 = clift.call %0(%u32) : !int32_to_float32
      // CHECK-NEXT: clift.yield %1 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NEXT: %0 = clift.use @int64_to_float32
      %0 = clift.use @int64_to_float32 : !int64_to_float32
      // CHECK-NEXT: %1 = clift.call %0(%arg0)
      %1 = clift.call %0(%i32) : !int64_to_float32
      // CHECK-NEXT: clift.yield %1 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      // CHECK-NEXT: %0 = clift.use @float64_lt
      %0 = clift.use @float64_lt : !float64_lt
      // CHECK-NEXT: %1 = clift.call %0(%arg2, %arg2)
      %1 = clift.call %0(%f32, %f32) : !float64_lt
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }
  }
}
