//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --promote-floating-point-operations | FileCheck %s

!void = !clift.void
!float32_t = !clift.float<4>
!float64_t = !clift.float<8>

!float32_add = !clift.func<
  "/helper-function/float32_add" as "float32_add" : !float32_t(!float32_t, !float32_t)
>

!float32_chs = !clift.func<
  "/helper-function/float32_chs" as "float32_chs" : !float32_t(!float32_t)
>

!float32_div = !clift.func<
  "/helper-function/float32_div" as "float32_div" : !float32_t(!float32_t, !float32_t)
>

!float32_mul = !clift.func<
  "/helper-function/float32_mul" as "float32_mul" : !float32_t(!float32_t, !float32_t)
>

!float32_sub = !clift.func<
  "/helper-function/float32_sub" as "float32_sub" : !float32_t(!float32_t, !float32_t)
>

!float64_add = !clift.func<
  "/helper-function/float64_add" as "float64_add" : !float64_t(!float64_t, !float64_t)
>

!float64_chs = !clift.func<
  "/helper-function/float64_chs" as "float64_chs" : !float64_t(!float64_t)
>

!float64_div = !clift.func<
  "/helper-function/float64_div" as "float64_div" : !float64_t(!float64_t, !float64_t)
>

!float64_mul = !clift.func<
  "/helper-function/float64_mul" as "float64_mul" : !float64_t(!float64_t, !float64_t)
>

!float64_sub = !clift.func<
  "/helper-function/float64_sub" as "float64_sub" : !float64_t(!float64_t, !float64_t)
>

!f = !clift.func<"" : !void(!float32_t, !float32_t, !float64_t, !float64_t)>

module attributes {clift.module} {
  clift.func @float32_add<!float32_add>(!float32_t, !float32_t)
  clift.func @float32_chs<!float32_chs>(!float32_t)
  clift.func @float32_div<!float32_div>(!float32_t, !float32_t)
  clift.func @float32_mul<!float32_mul>(!float32_t, !float32_t)
  clift.func @float32_sub<!float32_sub>(!float32_t, !float32_t)
  clift.func @float64_add<!float64_add>(!float64_t, !float64_t)
  clift.func @float64_chs<!float64_chs>(!float64_t)
  clift.func @float64_div<!float64_div>(!float64_t, !float64_t)
  clift.func @float64_mul<!float64_mul>(!float64_t, !float64_t)
  clift.func @float64_sub<!float64_sub>(!float64_t, !float64_t)

  // CHECK-LABEL: clift.func @f<
  clift.func @f<!f>(%a32 : !float32_t, %b32 : !float32_t, %a64 : !float64_t, %b64 : !float64_t) -> !void {
    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_add : !float32_add
      // CHECK-NEXT: %0 = clift.fadd %arg0, %arg1 : !float32_t
      %1 = clift.call %0(%a32, %b32) : !float32_add
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_chs : !float32_chs
      // CHECK-NEXT: %0 = clift.fneg %arg0 : !float32_t
      %1 = clift.call %0(%a32) : !float32_chs
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_div : !float32_div
      // CHECK-NEXT: %0 = clift.fdiv %arg0, %arg1 : !float32_t
      %1 = clift.call %0(%a32, %b32) : !float32_div
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_mul : !float32_mul
      // CHECK-NEXT: %0 = clift.fmul %arg0, %arg1 : !float32_t
      %1 = clift.call %0(%a32, %b32) : !float32_mul
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float32_sub : !float32_sub
      // CHECK-NEXT: %0 = clift.fsub %arg0, %arg1 : !float32_t
      %1 = clift.call %0(%a32, %b32) : !float32_sub
      // CHECK-NEXT: clift.yield %0 : !float32_t
      clift.yield %1 : !float32_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_add : !float64_add
      // CHECK-NEXT: %0 = clift.fadd %arg2, %arg3 : !float64_t
      %1 = clift.call %0(%a64, %b64) : !float64_add
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_chs : !float64_chs
      // CHECK-NEXT: %0 = clift.fneg %arg2 : !float64_t
      %1 = clift.call %0(%a64) : !float64_chs
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_div : !float64_div
      // CHECK-NEXT: %0 = clift.fdiv %arg2, %arg3 : !float64_t
      %1 = clift.call %0(%a64, %b64) : !float64_div
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_mul : !float64_mul
      // CHECK-NEXT: %0 = clift.fmul %arg2, %arg3 : !float64_t
      %1 = clift.call %0(%a64, %b64) : !float64_mul
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }

    // CHECK: clift.expr {
    clift.expr {
      %0 = clift.use @float64_sub : !float64_sub
      // CHECK-NEXT: %0 = clift.fsub %arg2, %arg3 : !float64_t
      %1 = clift.call %0(%a64, %b64) : !float64_sub
      // CHECK-NEXT: clift.yield %0 : !float64_t
      clift.yield %1 : !float64_t
    }
  }
}
