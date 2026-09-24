;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; RUN: %root/bin/revng opt -S -type-shrinking -verify %s | FileCheck %s

; A loop invariant needs a fixed point even though the eventual range is tiny.
declare i1 @again()
define i8 @cyclic_shift_amount(i64 %x, i64 %y, i3 %seed, i3 %mask, ptr %out) {
  ; CHECK-LABEL: @cyclic_shift_amount(
  ; CHECK: add i16
entry:
  %value = add i64 %x, %y
  %start = zext i3 %seed to i64
  %wide_mask = zext i3 %mask to i64
  br label %loop
loop:
  %amount = phi i64 [ %start, %entry ], [ %next, %loop ]
  %next = xor i64 %amount, %wide_mask
  %shifted = lshr i64 %value, %amount
  %low = trunc i64 %shifted to i8
  store i8 %low, ptr %out
  %continue = call i1 @again()
  br i1 %continue, label %loop, label %exit
exit:
  ret i8 %low
}

; An assumption on one successor cannot constrain the other successor.
declare void @llvm.assume(i1)
define i64 @local_assumption(i64 %x, i1 %c) {
  ; CHECK-LABEL: @local_assumption(
  ; CHECK: add i64
entry:
  %sum = add i64 %x, 1
  br i1 %c, label %then, label %else
then:
  %ok = icmp ult i64 %sum, 256
  call void @llvm.assume(i1 %ok)
  ret i64 %sum
else:
  ret i64 %sum
}

; The signed-byte range of a value cycling through an XOR needs a fixed point.
define i64 @signed_xor_cycle(i8 %seed, i8 %mask) {
  ; CHECK-LABEL: @signed_xor_cycle(
  ; CHECK: phi i8
  ; CHECK: xor i8
entry:
  %start = sext i8 %seed to i64
  %wide_mask = sext i8 %mask to i64
  br label %loop
loop:
  %value = phi i64 [ %start, %entry ], [ %next, %loop ]
  %next = xor i64 %value, %wide_mask
  %continue = call i1 @again()
  br i1 %continue, label %loop, label %exit
exit:
  ret i64 %next
}

; SCCP leaves the infeasible successor unvisited, although CFG traversal can
; still reach it. Querying its values must not require a solver state.
define i64 @infeasible_successor(i64 %x) {
  ; CHECK-LABEL: @infeasible_successor(
  ; CHECK: ret i64 %x
  ; CHECK: add i64
entry:
  br i1 true, label %taken, label %untaken
taken:
  ret i64 %x
untaken:
  %sum = add i64 %x, 1
  ret i64 %sum
}
