;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -verify-loop-lcssa -loop-rewrite-with-canonical-induction-variable  %s -S -o - | FileCheck %s

; Three nested loops. `middle.latch` is the exit block of the innermost loop,
; so it holds an LCSSA phi for `%inner.value`, and it is at the same time a
; block of the middle loop.
;
; After rewriting the induction variables the pass cleans up by running
; `SimplifyInstructionsInBlock` over every block of the loop it ran on. That
; reaches `middle.latch`, where the LCSSA phi has a single incoming value and
; so folds away, taking the innermost loop out of LCSSA form.
; To avoid this, we have to make sure the pass rebuilds LCSSA for the loops
; whose exit blocks it simplifies.
;
; `-verify-loop-lcssa` makes the loop pass manager check
; `isRecursivelyLCSSAForm` after every loop pass. This catches the inner loop
; not being in LCSSA. This is not strictly necessary, since dropping this
; verification pass causes a crash, when running on the inner loop anyway, but
; it improves diagnostic for every potential test failure.


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @use(i64)

define void @nested_loops(i1 %condition) {
entry:
  br label %outer

outer:
  %i = phi i64 [ 0, %entry ], [ %i.next, %middle.latch ]
  %i.next = add i64 %i, 1
  br label %middle

middle:
  %j = phi i64 [ 0, %outer ], [ %j.next, %middle.latch ]
  %j.plus = add i64 %j, 1
  br i1 %condition, label %inner, label %middle.latch

inner:
  %inner.value = add i64 %i, 1
  call void @use(i64 %j.plus)
  br i1 %condition, label %inner, label %middle.latch

middle.latch:
  %inner.value.lcssa = phi i64 [ 0, %middle ], [ %inner.value, %inner ]
  call void @use(i64 %inner.value.lcssa)
  %j.next = add i64 %j, 1
  br i1 %condition, label %outer, label %middle
}

; Producing anything at all is the point: if without the repair the pass asserts
; inside `getInsertPointForUses` and emits nothing.

; CHECK: define void @nested_loops
; CHECK: call void @use
