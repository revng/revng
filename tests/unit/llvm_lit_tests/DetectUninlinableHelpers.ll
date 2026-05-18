;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; Check that `detect-uninlinable-helpers` classifies
; every `revng_inline` helper into one of three buckets and encodes
; the result on the function itself: always-inline and conditional-on-
; constant-args via `!revng.inline.policy` metadata (an `iN` bitmask
; over the formal-parameter set; `0` means "always"), never-inline by
; stripping the `revng_inline` section attribute.
;
; RUN: %revngopt -detect-uninlinable-helpers %s -S -o - | FileCheck %s

!qemu.architecture = !{!0}
!0 = !{!"x86_64"}

; A `setConstant(true)` global with an initializer: a direct load
; through it folds at post-inline time (same semantics as `@env`
; after `VariableManager: mark @env as constant`).
@const_int = constant i32 7

; A writable global representing a runtime CSV; loads through it
; are genuinely runtime and short-circuit the kernel to nullopt.
@runtime_csv = global i32 0

;
; (1) `always`: no switch and no `getelementptr` index depends on
; any formal parameter; no runtime-memory load on any chain.
;
define i64 @helper_always(i64 %x) section "revng_inline" {
entry:
  %r = add i64 %x, 42
  ret i64 %r
}
; CHECK-DAG: define i64 @helper_always(i64 %x) section "revng_inline" !revng.inline.policy ![[ALWAYS:[0-9]+]]

;
; (2) `conditional` from a `switch` directly on an argument.
;
define i64 @helper_conditional_switch(i64 %x, i32 %op) section "revng_inline" {
entry:
  switch i32 %op, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]
case0:
  ret i64 %x
case1:
  %inc = add i64 %x, 1
  ret i64 %inc
default:
  ret i64 0
}
; CHECK-DAG: define i64 @helper_conditional_switch(i64 %x, i32 %op) section "revng_inline" !revng.inline.policy ![[COND_SWITCH:[0-9]+]]

;
; (3) `conditional` from a `getelementptr` indexed by an argument
; (no switch on a runtime-memory load on the chain). The kernel
; treats GEP indices as critical operands too.
;
define ptr @helper_conditional_gep(i64 %idx, ptr %base) section "revng_inline" {
entry:
  %p = getelementptr i64, ptr %base, i64 %idx
  ret ptr %p
}
; CHECK-DAG: define ptr @helper_conditional_gep(i64 %idx, ptr %base) section "revng_inline" !revng.inline.policy ![[COND_GEP:[0-9]+]]

;
; (4) `always` even with a `switch`, because the switch's
; condition reaches a load through a `setConstant(true)` global
; that has an initializer — that load folds at post-inline time.
;
define i64 @helper_always_load_const(i64 %x) section "revng_inline" {
entry:
  %v = load i32, ptr @const_int
  switch i32 %v, label %default [
    i32 7, label %case7
  ]
case7:
  ret i64 %x
default:
  ret i64 0
}
; CHECK-DAG: define i64 @helper_always_load_const(i64 %x) section "revng_inline" !revng.inline.policy ![[ALWAYS]]

;
; (5) never-inline (load-from-runtime-memory on a switch chain):
; the switch's condition reaches a load from a writable global..
; The static pass must strip the `revng_inline` section attribute on this
; helper.
;
define i64 @helper_never_runtime_load(i64 %x) section "revng_inline" {
entry:
  %op = load i32, ptr @runtime_csv
  switch i32 %op, label %default [
    i32 0, label %case0
  ]
case0:
  ret i64 %x
default:
  ret i64 0
}
; CHECK-NOT: define {{.*}}@helper_never_runtime_load{{.*}}section "revng_inline"

;
; (6) never-inline (cyclic SCC member): two helpers calling each
; other form a cyclic SCC of the call graph, so neither can be
; inlined without an unbounded fixed-point loop.
;
define i64 @helper_recursive_a(i64 %x) section "revng_inline" {
entry:
  %r = call i64 @helper_recursive_b(i64 %x)
  ret i64 %r
}
; CHECK-NOT: define {{.*}}@helper_recursive_a{{.*}}section "revng_inline"

define i64 @helper_recursive_b(i64 %x) section "revng_inline" {
entry:
  %r = call i64 @helper_recursive_a(i64 %x)
  ret i64 %r
}
; CHECK-NOT: define {{.*}}@helper_recursive_b{{.*}}section "revng_inline"

; Checks for the metadata shapes themselves.
; CHECK-DAG: ![[ALWAYS]] = !{i2 0}
; CHECK-DAG: ![[COND_SWITCH]] = !{i3 2}
; CHECK-DAG: ![[COND_GEP]] = !{i3 1}
