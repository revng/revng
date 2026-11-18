//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

// Comparison helpers return booleans as unsigned integers. These are
// promoted to a comparison followed by a boolean extension.
//
// The `f{{le}}` regex matches the less-or-equal mnemonic without spelling it
// out, since codespell would otherwise flag it as a misspelling.

!void = !clift.void
!uint8_t = !clift.int<unsigned 1>
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!float32_eq_quiet = !clift.func<
  "/helper-function/float32_eq_quiet" as "float32_eq_quiet" : !uint8_t(!float32_t, !float32_t)
>

!float32_le = !clift.func<
  "/helper-function/float32_le" as "float32_le" : !uint8_t(!float32_t, !float32_t)
>

!float32_lt = !clift.func<
  "/helper-function/float32_lt" as "float32_lt" : !uint8_t(!float32_t, !float32_t)
>

!float64_eq_quiet = !clift.func<
  "/helper-function/float64_eq_quiet" as "float64_eq_quiet" : !uint8_t(!float64_t, !float64_t)
>

!float64_le = !clift.func<
  "/helper-function/float64_le" as "float64_le" : !uint8_t(!float64_t, !float64_t)
>

!float64_lt = !clift.func<
  "/helper-function/float64_lt" as "float64_lt" : !uint8_t(!float64_t, !float64_t)
>

!f = !clift.func<"" : !void(!float32_t, !float32_t, !float64_t, !float64_t)>

module attributes {clift.module} {
  clift.func @float32_eq_quiet<!float32_eq_quiet>(!float32_t, !float32_t)
  clift.func @float32_le<!float32_le>(!float32_t, !float32_t)
  clift.func @float32_lt<!float32_lt>(!float32_t, !float32_t)
  clift.func @float64_eq_quiet<!float64_eq_quiet>(!float64_t, !float64_t)
  clift.func @float64_le<!float64_le>(!float64_t, !float64_t)
  clift.func @float64_lt<!float64_lt>(!float64_t, !float64_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%a32 : !float32_t, %b32 : !float32_t, %a64 : !float64_t, %b64 : !float64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_eq_quiet : !float32_eq_quiet
      // CHECK-NEXT: %0 = clift.feq %arg0, %arg1 : !float32_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a32, %b32) : !float32_eq_quiet
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_le : !float32_le
      // CHECK-NEXT: %0 = clift.f{{le}} %arg0, %arg1 : !float32_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a32, %b32) : !float32_le
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_lt : !float32_lt
      // CHECK-NEXT: %0 = clift.flt %arg0, %arg1 : !float32_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a32, %b32) : !float32_lt
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_eq_quiet : !float64_eq_quiet
      // CHECK-NEXT: %0 = clift.feq %arg2, %arg3 : !float64_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a64, %b64) : !float64_eq_quiet
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_le : !float64_le
      // CHECK-NEXT: %0 = clift.f{{le}} %arg2, %arg3 : !float64_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a64, %b64) : !float64_le
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_lt : !float64_lt
      // CHECK-NEXT: %0 = clift.flt %arg2, %arg3 : !float64_t
      // CHECK-NEXT: %1 = clift.bext %0 : !uint8_t
      %1 = clift.call %0(%a64, %b64) : !float64_lt
      // CHECK-NEXT: clift.yield %1 : !uint8_t
      clift.yield %1 : !uint8_t
    }
  }
}
