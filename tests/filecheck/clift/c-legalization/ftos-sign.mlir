//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --c-legalization | FileCheck %s

!void = !clift.void
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>
!int32_t = !clift.int<signed 4>
!int64_t = !clift.int<signed 8>
!uint32_t = !clift.int<unsigned 4>
!uint64_t = !clift.int<unsigned 8>

!f = !clift.func<
  "/type-definition/1001-CABIFunctionDefinition" : !void(!float32_t, !float64_t)
>

module attributes {clift.module} {
  clift.func @f<!f>(%f32 : !float32_t, %f64 : !float64_t) -> !void attributes {
    handle = "/function/0x40001001:Code_x86_64"
  } {
    // CHECK: %0 = clift.local : !int32_t
    %i32 = clift.local : !int32_t
    // CHECK: %1 = clift.local : !uint32_t
    %u32 = clift.local : !uint32_t
    // CHECK: %2 = clift.local : !uint64_t
    %u64 = clift.local : !uint64_t

    // A result that is already signed is left alone.
    clift.expr {
      // CHECK: %3 = clift.ftos %arg1 : !float64_t -> !int32_t
      %0 = clift.ftos %f64 : !float64_t -> !int32_t
      // CHECK-NEXT: %4 = clift.assign %0, %3 : !int32_t
      %1 = clift.assign %i32, %0 : !int32_t
      // CHECK-NEXT: clift.yield %4 : !int32_t
      clift.yield %1 : !int32_t
    }

    // An unsigned result is made signed, and cast back for its user.
    clift.expr {
      // CHECK: %3 = clift.ftos %arg1 : !float64_t -> !int32_t
      %0 = clift.ftos %f64 : !float64_t -> !uint32_t
      // CHECK-NEXT: %4 = clift.bitcast %3 : !int32_t -> !uint32_t
      // CHECK-NEXT: %5 = clift.assign %1, %4 : !uint32_t
      %1 = clift.assign %u32, %0 : !uint32_t
      // CHECK-NEXT: clift.yield %5 : !uint32_t
      clift.yield %1 : !uint32_t
    }

    // The signed type is taken from the result, not from the operand, so the
    // result keeps its own width even where it differs from the float's.
    clift.expr {
      // CHECK: %3 = clift.ftos %arg0 : !float32_t -> !int64_t
      %0 = clift.ftos %f32 : !float32_t -> !uint64_t
      // CHECK-NEXT: %4 = clift.bitcast %3 : !int64_t -> !uint64_t
      // CHECK-NEXT: %5 = clift.assign %2, %4 : !uint64_t
      %1 = clift.assign %u64, %0 : !uint64_t
      // CHECK-NEXT: clift.yield %5 : !uint64_t
      clift.yield %1 : !uint64_t
    }
  }
}
