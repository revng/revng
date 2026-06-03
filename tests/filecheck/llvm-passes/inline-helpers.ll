;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; Check that `inline-helpers` consumes the `!revng.inline.policy`
; metadata attached by `detect-uninlinable-helpers`
; at helpers-build time and uses it to gate per-call-site inlining.
; A helper with a non-zero policy must only be inlined at call sites
; where every listed argument is a compile-time constant. A helper
; with a zero policy must always be inlined.
;
; RUN: %root/bin/revng opt -inline-helpers %s -S -o - | FileCheck %s

!revng.qemu_architecture = !{!0}
!0 = !{!"x86_64"}

; A helper with a source-level switch on arg #1 - the static pass marked
; arg #1 as the single critical operand (bit 1 set in `i2 2`).
;
; Once `inline-helpers` has finished its per-isolated-function pass, the
; helper's body is blanked: the dynamic-op call site is left for the
; linker, so the helper survives as a declaration.
; CHECK: declare {{.*}}@helper_with_critical
define i64 @helper_with_critical(i64 %x, i32 %op) section "revng_inline" !revng.inline.policy !20 {
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

; A helper with no critical operands — the static pass produced the
; empty BitVector, encoded as `i1 0`. Since all its calls get inlined
; away, the body is blanked and it ends up as a declaration too.
; CHECK: declare {{.*}}@helper_no_critical
define i64 @helper_no_critical(i64 %x) section "revng_inline" !revng.inline.policy !21 {
entry:
  %r = add i64 %x, 42
  ret i64 %r
}

; An Isolated function with three calls to the helpers above: two
; call sites of `@helper_with_critical` - one with a constant `op`
; (inlineable) and one with a dynamic `op` (not inlineable) - plus
; one call to `@helper_no_critical` (always inlined).
; CHECK-LABEL: define i64 @my_isolated_function(i64 %v, i32 %op)
define i64 @my_isolated_function(i64 %v, i32 %op) !revng.tags !10 {
entry:
  ; The constant-op call site must be inlined away.
  ; CHECK-NOT: call i64 @helper_with_critical(i64 %v, i32 0)
  %a = call i64 @helper_with_critical(i64 %v, i32 0)
  ; The dynamic-op call site must survive.
  ; CHECK: call i64 @helper_with_critical(i64 %v, i32 %op)
  %b = call i64 @helper_with_critical(i64 %v, i32 %op)
  ; The no-critical-args call must be inlined away.
  ; CHECK-NOT: call i64 @helper_no_critical
  %c = call i64 @helper_no_critical(i64 %v)
  %ab = add i64 %a, %b
  %abc = add i64 %ab, %c
  ret i64 %abc
}

!10 = !{!"isolated"}
!20 = !{i3 2}
!21 = !{i2 0}
