;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -peephole-opt-for-decompilation %s -S -o - | FileCheck %s

;
; CHECK-LABEL: define i64 @add_eq()
;
; The compare on %counter is rewritten to reuse %next (%counter + 1), adjusting
; the constant accordingly (%counter == 4 becomes %next == 5).
define i64 @add_eq() {
; CHECK: entry:
entry:
  br label %loop

; CHECK: loop:
loop:
  %counter = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = add i64 %counter, 1
  %next = add i64 %counter, 1
  ; CHECK-NEXT: [[COND:%[a-zA-Z_]+]] = icmp eq i64 [[NEXT]], 5
  %cond = icmp eq i64 %counter, 4
  ; CHECK-NEXT: br i1 [[COND]], label %exit, label %loop
  br i1 %cond, label %exit, label %loop

; CHECK: exit:
exit:
  ret i64 %next
}

;
; CHECK-LABEL: define i64 @add_neq()
;
define i64 @add_neq() {
; CHECK: entry:
entry:
  br label %loop

; CHECK: loop:
loop:
  %counter = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = add i64 %counter, 1
  %next = add i64 %counter, 1
  ; CHECK-NEXT: [[COND:%[a-zA-Z_]+]] = icmp ne i64 [[NEXT]], 5
  %cond = icmp ne i64 %counter, 4
  ; CHECK-NEXT: br i1 [[COND]], label %exit, label %loop
  br i1 %cond, label %exit, label %loop

; CHECK: exit:
exit:
  ret i64 %next
}

;
; CHECK-LABEL: define i64 @sub_eq()
;
define i64 @sub_eq() {
; CHECK: entry:
entry:
  br label %loop

; CHECK: loop:
loop:
  %counter = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = sub i64 %counter, 1
  %next = sub i64 %counter, 1
  ; CHECK-NEXT: [[COND:%[a-zA-Z_]+]] = icmp eq i64 [[NEXT]], 3
  %cond = icmp eq i64 %counter, 4
  ; CHECK-NEXT: br i1 [[COND]], label %exit, label %loop
  br i1 %cond, label %exit, label %loop

; CHECK: exit:
exit:
  ret i64 %next
}

;
; CHECK-LABEL: define i64 @sub_neq()
;
define i64 @sub_neq() {
; CHECK: entry:
entry:
  br label %loop

; CHECK: loop:
loop:
  %counter = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = sub i64 %counter, 1
  %next = sub i64 %counter, 1
  ; CHECK-NEXT: [[COND:%[a-zA-Z_]+]] = icmp ne i64 [[NEXT]], 3
  %cond = icmp ne i64 %counter, 4
  ; CHECK-NEXT: br i1 [[COND]], label %exit, label %loop
  br i1 %cond, label %exit, label %loop

; CHECK: exit:
exit:
  ret i64 %next
}

;
; CHECK-LABEL: define i64 @reorder()
;
; The compare comes before %next in program order, so %next does not yet
; dominate it. The pass hoists %next above the compare and then rewrites it.
define i64 @reorder() {
; CHECK: entry:
entry:
  br label %loop

; CHECK: loop:
loop:
  %counter = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = sub i64 %counter, 1
  ; CHECK-NEXT: [[COND:%[a-zA-Z_]+]] = icmp ne i64 [[NEXT]], 3
  %cond = icmp ne i64 %counter, 4
  %next = sub i64 %counter, 1
  ; CHECK-NEXT: br i1 [[COND]], label %exit, label %loop
  br i1 %cond, label %exit, label %loop

; CHECK: exit:
exit:
  ret i64 %next
}

;
; CHECK-LABEL: define void @reorder_2()
;
; %next is defined in %body but is an incoming of the %header PHI. %headcond
; dominates %next, so the pass hoists %next before it (into %header) and
; rewrites the compares it dominates (%headcond and %bodycond). %othercond is
; left unchanged: %other and %body are siblings, neither dominates the other.
define void @reorder_2() {
; CHECK: entry:
entry:
  br label %header

; CHECK: header:
header:
  %counter = phi i32 [ 3, %entry ], [ %next, %body ]
  ; CHECK: [[NEXT:%[a-zA-Z_]+]] = add i32 %counter, 1
  ; CHECK-NEXT: icmp ne i32 [[NEXT]], 11
  %headcond = icmp ne i32 %counter, 10
  br i1 %headcond, label %body, label %other

; CHECK: body:
body:
  %next = add i32 %counter, 1
  ; CHECK: icmp ne i32 [[NEXT]], 8
  %bodycond = icmp ne i32 %counter, 7
  br i1 %bodycond, label %exit, label %header

; CHECK: other:
other:
  ; The sibling compare must be left untouched (not rewritten to use [[NEXT]]).
  ; CHECK-NOT: [[NEXT]]
  %othercond = icmp eq i32 %counter, 3
  br label %exit

; CHECK: exit:
exit:
  ret void
}
