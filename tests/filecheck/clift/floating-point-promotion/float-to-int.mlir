//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

!void = !clift.void
!int16_t = !clift.int<signed 2>
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>
!uint16_t = !clift.int<unsigned 2>
!uint32_t = !clift.int<unsigned 4>
!uint64_t = !clift.int<unsigned 8>
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!float32_to_int16_round_to_zero = !clift.func<
  "/helper-function/float32_to_int16_round_to_zero" as "float32_to_int16_round_to_zero" : !int16_t(!float32_t)
>

!float32_to_int32_round_to_zero = !clift.func<
  "/helper-function/float32_to_int32_round_to_zero" as "float32_to_int32_round_to_zero" : !int32_t(!float32_t)
>

!float32_to_int64_round_to_zero = !clift.func<
  "/helper-function/float32_to_int64_round_to_zero" as "float32_to_int64_round_to_zero" : !int64_t(!float32_t)
>

!float32_to_uint16_round_to_zero = !clift.func<
  "/helper-function/float32_to_uint16_round_to_zero" as "float32_to_uint16_round_to_zero" : !uint16_t(!float32_t)
>

!float32_to_uint32_round_to_zero = !clift.func<
  "/helper-function/float32_to_uint32_round_to_zero" as "float32_to_uint32_round_to_zero" : !uint32_t(!float32_t)
>

!float32_to_uint64_round_to_zero = !clift.func<
  "/helper-function/float32_to_uint64_round_to_zero" as "float32_to_uint64_round_to_zero" : !uint64_t(!float32_t)
>

!float64_to_int16_round_to_zero = !clift.func<
  "/helper-function/float64_to_int16_round_to_zero" as "float64_to_int16_round_to_zero" : !int16_t(!float64_t)
>

!float64_to_int32_round_to_zero = !clift.func<
  "/helper-function/float64_to_int32_round_to_zero" as "float64_to_int32_round_to_zero" : !int32_t(!float64_t)
>

!float64_to_int64_round_to_zero = !clift.func<
  "/helper-function/float64_to_int64_round_to_zero" as "float64_to_int64_round_to_zero" : !int64_t(!float64_t)
>

!float64_to_uint16_round_to_zero = !clift.func<
  "/helper-function/float64_to_uint16_round_to_zero" as "float64_to_uint16_round_to_zero" : !uint16_t(!float64_t)
>

!float64_to_uint32_round_to_zero = !clift.func<
  "/helper-function/float64_to_uint32_round_to_zero" as "float64_to_uint32_round_to_zero" : !uint32_t(!float64_t)
>

!float64_to_uint64_round_to_zero = !clift.func<
  "/helper-function/float64_to_uint64_round_to_zero" as "float64_to_uint64_round_to_zero" : !uint64_t(!float64_t)
>

!f = !clift.func<"" : !void(!float32_t, !float64_t)>

module attributes {clift.module} {
  clift.func @float32_to_int16_round_to_zero<!float32_to_int16_round_to_zero>(!float32_t)
  clift.func @float32_to_int32_round_to_zero<!float32_to_int32_round_to_zero>(!float32_t)
  clift.func @float32_to_int64_round_to_zero<!float32_to_int64_round_to_zero>(!float32_t)
  clift.func @float32_to_uint16_round_to_zero<!float32_to_uint16_round_to_zero>(!float32_t)
  clift.func @float32_to_uint32_round_to_zero<!float32_to_uint32_round_to_zero>(!float32_t)
  clift.func @float32_to_uint64_round_to_zero<!float32_to_uint64_round_to_zero>(!float32_t)
  clift.func @float64_to_int16_round_to_zero<!float64_to_int16_round_to_zero>(!float64_t)
  clift.func @float64_to_int32_round_to_zero<!float64_to_int32_round_to_zero>(!float64_t)
  clift.func @float64_to_int64_round_to_zero<!float64_to_int64_round_to_zero>(!float64_t)
  clift.func @float64_to_uint16_round_to_zero<!float64_to_uint16_round_to_zero>(!float64_t)
  clift.func @float64_to_uint32_round_to_zero<!float64_to_uint32_round_to_zero>(!float64_t)
  clift.func @float64_to_uint64_round_to_zero<!float64_to_uint64_round_to_zero>(!float64_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%f32 : !float32_t, %f64 : !float64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_int16_round_to_zero : !float32_to_int16_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg0 : !float32_t -> !int16_t
      %1 = clift.call %0(%f32) : !float32_to_int16_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int16_t
      clift.yield %1 : !int16_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_int32_round_to_zero : !float32_to_int32_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg0 : !float32_t -> !int32_t
      %1 = clift.call %0(%f32) : !float32_to_int32_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int32_t
      clift.yield %1 : !int32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_int64_round_to_zero : !float32_to_int64_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg0 : !float32_t -> !int64_t
      %1 = clift.call %0(%f32) : !float32_to_int64_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int64_t
      clift.yield %1 : !int64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_uint16_round_to_zero : !float32_to_uint16_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg0 : !float32_t -> !uint16_t
      %1 = clift.call %0(%f32) : !float32_to_uint16_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint16_t
      clift.yield %1 : !uint16_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_uint32_round_to_zero : !float32_to_uint32_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg0 : !float32_t -> !uint32_t
      %1 = clift.call %0(%f32) : !float32_to_uint32_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint32_t
      clift.yield %1 : !uint32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_to_uint64_round_to_zero : !float32_to_uint64_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg0 : !float32_t -> !uint64_t
      %1 = clift.call %0(%f32) : !float32_to_uint64_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint64_t
      clift.yield %1 : !uint64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_int16_round_to_zero : !float64_to_int16_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg1 : !float64_t -> !int16_t
      %1 = clift.call %0(%f64) : !float64_to_int16_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int16_t
      clift.yield %1 : !int16_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_int32_round_to_zero : !float64_to_int32_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg1 : !float64_t -> !int32_t
      %1 = clift.call %0(%f64) : !float64_to_int32_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int32_t
      clift.yield %1 : !int32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_int64_round_to_zero : !float64_to_int64_round_to_zero
      // CHECK-NEXT: %0 = clift.ftos %arg1 : !float64_t -> !int64_t
      %1 = clift.call %0(%f64) : !float64_to_int64_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !int64_t
      clift.yield %1 : !int64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_uint16_round_to_zero : !float64_to_uint16_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg1 : !float64_t -> !uint16_t
      %1 = clift.call %0(%f64) : !float64_to_uint16_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint16_t
      clift.yield %1 : !uint16_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_uint32_round_to_zero : !float64_to_uint32_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg1 : !float64_t -> !uint32_t
      %1 = clift.call %0(%f64) : !float64_to_uint32_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint32_t
      clift.yield %1 : !uint32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_to_uint64_round_to_zero : !float64_to_uint64_round_to_zero
      // CHECK-NEXT: %0 = clift.ftou %arg1 : !float64_t -> !uint64_t
      %1 = clift.call %0(%f64) : !float64_to_uint64_round_to_zero
      // CHECK-NEXT: clift.yield %0 : !uint64_t
      clift.yield %1 : !uint64_t
    }
  }
}
