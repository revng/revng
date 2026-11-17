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

!int16_to_float32 = !clift.func<
  "/helper-function/int16_to_float32" as "int16_to_float32" : !float32_t(!int16_t)
>

!int16_to_float64 = !clift.func<
  "/helper-function/int16_to_float64" as "int16_to_float64" : !float64_t(!int16_t)
>

!int32_to_float32 = !clift.func<
  "/helper-function/int32_to_float32" as "int32_to_float32" : !float32_t(!int32_t)
>

!int32_to_float64 = !clift.func<
  "/helper-function/int32_to_float64" as "int32_to_float64" : !float64_t(!int32_t)
>

!int64_to_float32 = !clift.func<
  "/helper-function/int64_to_float32" as "int64_to_float32" : !float32_t(!int64_t)
>

!int64_to_float64 = !clift.func<
  "/helper-function/int64_to_float64" as "int64_to_float64" : !float64_t(!int64_t)
>

!uint16_to_float32 = !clift.func<
  "/helper-function/uint16_to_float32" as "uint16_to_float32" : !float32_t(!uint16_t)
>

!uint16_to_float64 = !clift.func<
  "/helper-function/uint16_to_float64" as "uint16_to_float64" : !float64_t(!uint16_t)
>

!uint32_to_float32 = !clift.func<
  "/helper-function/uint32_to_float32" as "uint32_to_float32" : !float32_t(!uint32_t)
>

!uint32_to_float64 = !clift.func<
  "/helper-function/uint32_to_float64" as "uint32_to_float64" : !float64_t(!uint32_t)
>

!uint64_to_float32 = !clift.func<
  "/helper-function/uint64_to_float32" as "uint64_to_float32" : !float32_t(!uint64_t)
>

!uint64_to_float64 = !clift.func<
  "/helper-function/uint64_to_float64" as "uint64_to_float64" : !float64_t(!uint64_t)
>

!f = !clift.func<"" : !void(!int16_t, !int32_t, !int64_t, !uint16_t, !uint32_t, !uint64_t)>

module attributes {clift.module} {
  clift.func @int16_to_float32<!int16_to_float32>(!int16_t)
  clift.func @int16_to_float64<!int16_to_float64>(!int16_t)
  clift.func @int32_to_float32<!int32_to_float32>(!int32_t)
  clift.func @int32_to_float64<!int32_to_float64>(!int32_t)
  clift.func @int64_to_float32<!int64_to_float32>(!int64_t)
  clift.func @int64_to_float64<!int64_to_float64>(!int64_t)
  clift.func @uint16_to_float32<!uint16_to_float32>(!uint16_t)
  clift.func @uint16_to_float64<!uint16_to_float64>(!uint16_t)
  clift.func @uint32_to_float32<!uint32_to_float32>(!uint32_t)
  clift.func @uint32_to_float64<!uint32_to_float64>(!uint32_t)
  clift.func @uint64_to_float32<!uint64_to_float32>(!uint64_t)
  clift.func @uint64_to_float64<!uint64_to_float64>(!uint64_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%i16 : !int16_t, %i32 : !int32_t, %i64 : !int64_t, %u16 : !uint16_t, %u32 : !uint32_t, %u64 : !uint64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int16_to_float32 : !int16_to_float32
      // CHECK-NEXT: %0 = clift.stof %arg0 : !int16_t -> !float32_t
      %1 = clift.call %0(%i16) : !int16_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int16_to_float64 : !int16_to_float64
      // CHECK-NEXT: %0 = clift.stof %arg0 : !int16_t -> !float64_t
      %1 = clift.call %0(%i16) : !int16_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int32_to_float32 : !int32_to_float32
      // CHECK-NEXT: %0 = clift.stof %arg1 : !int32_t -> !float32_t
      %1 = clift.call %0(%i32) : !int32_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int32_to_float64 : !int32_to_float64
      // CHECK-NEXT: %0 = clift.stof %arg1 : !int32_t -> !float64_t
      %1 = clift.call %0(%i32) : !int32_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int64_to_float32 : !int64_to_float32
      // CHECK-NEXT: %0 = clift.stof %arg2 : !int64_t -> !float32_t
      %1 = clift.call %0(%i64) : !int64_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @int64_to_float64 : !int64_to_float64
      // CHECK-NEXT: %0 = clift.stof %arg2 : !int64_t -> !float64_t
      %1 = clift.call %0(%i64) : !int64_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint16_to_float32 : !uint16_to_float32
      // CHECK-NEXT: %0 = clift.utof %arg3 : !uint16_t -> !float32_t
      %1 = clift.call %0(%u16) : !uint16_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint16_to_float64 : !uint16_to_float64
      // CHECK-NEXT: %0 = clift.utof %arg3 : !uint16_t -> !float64_t
      %1 = clift.call %0(%u16) : !uint16_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint32_to_float32 : !uint32_to_float32
      // CHECK-NEXT: %0 = clift.utof %arg4 : !uint32_t -> !float32_t
      %1 = clift.call %0(%u32) : !uint32_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint32_to_float64 : !uint32_to_float64
      // CHECK-NEXT: %0 = clift.utof %arg4 : !uint32_t -> !float64_t
      %1 = clift.call %0(%u32) : !uint32_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint64_to_float32 : !uint64_to_float32
      // CHECK-NEXT: %0 = clift.utof %arg5 : !uint64_t -> !float32_t
      %1 = clift.call %0(%u64) : !uint64_to_float32
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @uint64_to_float64 : !uint64_to_float64
      // CHECK-NEXT: %0 = clift.utof %arg5 : !uint64_t -> !float64_t
      %1 = clift.call %0(%u64) : !uint64_to_float64
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }
  }
}
