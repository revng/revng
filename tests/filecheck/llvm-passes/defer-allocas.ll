;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -defer-allocas -S -o - | FileCheck %s

;
; CHECK-LABEL: define i64 @used_in_branch(i1 %c)
;
; All the accesses are in Then: the alloca moves there.
define i64 @used_in_branch(i1 %c) {
Entry:
  %variable = alloca i64
  br i1 %c, label %Then, label %Exit

; CHECK: Then:
; CHECK-NEXT: %variable = alloca i64
; CHECK-NEXT: store i64 1, ptr %variable
Then:
  store i64 1, ptr %variable
  %value = load i64, ptr %variable
  br label %Exit

Exit:
  %result = phi i64 [ %value, %Then ], [ 0, %Entry ]
  ret i64 %result
}

;
; CHECK-LABEL: define i64 @carried_across_iterations(i64 %n)
;
; The variable is undefined when entering Loop, which reads it at its beginning
; and writes it for the next iteration. The store of undef in Entry keeps the
; alloca before the loop, and is then dropped.
define i64 @carried_across_iterations(i64 %n) {
; CHECK: Entry:
; CHECK-NEXT: %variable = alloca i64
; CHECK-NEXT: br label %Loop
Entry:
  %variable = alloca i64
  store i64 undef, ptr %variable
  br label %Loop

; CHECK: Loop:
; CHECK-NOT: alloca
; CHECK: ret
Loop:
  %old = load i64, ptr %variable
  %new = add i64 %old, %n
  store i64 %new, ptr %variable
  %done = icmp eq i64 %new, 100
  br i1 %done, label %Exit, label %Loop

Exit:
  ret i64 %new
}

;
; CHECK-LABEL: define i64 @written_before_read_in_loop(i64 %n)
;
; The variable is written before being read in every iteration: it can be
; declared in the loop.
define i64 @written_before_read_in_loop(i64 %n) {
Entry:
  %variable = alloca i64
  br label %Loop

; CHECK: Loop:
; CHECK-NEXT: %variable = alloca i64
; CHECK-NEXT: store i64 %n, ptr %variable
Loop:
  store i64 %n, ptr %variable
  %value = load i64, ptr %variable
  %done = icmp eq i64 %value, 100
  br i1 %done, label %Exit, label %Loop

Exit:
  ret i64 %value
}

;
; CHECK-LABEL: define i64 @carried_in_irreducible_cycle(i1 %c, i64 %n)
;
; The cycle First -> Second -> First can be entered from both blocks, and First
; reads the variable before writing it. The store of undef in Entry keeps the
; alloca before the cycle, and is then dropped.
define i64 @carried_in_irreducible_cycle(i1 %c, i64 %n) {
; CHECK: Entry:
; CHECK-NEXT: %variable = alloca i64
; CHECK-NEXT: br i1 %c, label %First, label %Second
Entry:
  %variable = alloca i64
  store i64 undef, ptr %variable
  br i1 %c, label %First, label %Second

; CHECK: First:
; CHECK-NOT: alloca
; CHECK: ret
First:
  %old = load i64, ptr %variable
  %new = add i64 %old, %n
  store i64 %new, ptr %variable
  %done = icmp eq i64 %new, 100
  br i1 %done, label %Exit, label %Second

Second:
  br label %First

Exit:
  ret i64 %new
}

;
; CHECK-LABEL: define i64 @assigned_in_branches_in_loop(i64 %n, i1 %c)
;
; Both branches write the variable before Latch reads it: it can be declared
; in the loop.
define i64 @assigned_in_branches_in_loop(i64 %n, i1 %c) {
Entry:
  %variable = alloca i64
  br label %Loop

; CHECK: Loop:
; CHECK-NEXT: %variable = alloca i64
; CHECK-NEXT: br i1 %c, label %Then, label %Else
Loop:
  br i1 %c, label %Then, label %Else

Then:
  store i64 1, ptr %variable
  br label %Latch

Else:
  store i64 2, ptr %variable
  br label %Latch

Latch:
  %value = load i64, ptr %variable
  %done = icmp eq i64 %value, %n
  br i1 %done, label %Exit, label %Loop

Exit:
  ret i64 %value
}
