;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt -peephole-opt-for-decompilation %s -S -o - | FileCheck %s

define i64 @add_eq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rax.0 = phi i64 [ 0, %Entry ], [ %2, %Loop ]
  %rcx.0 = phi i64 [ 0, %Entry ], [ %3, %Loop ]
  %1 = load i64, ptr %0, align 1
  %2 = add i64 %rax.0, %1
  %3 = add nuw nsw i64 %rcx.0, 1
  ; CHECK: %4 = icmp eq i64 %3, 5
  %4 = icmp eq i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %4, label %Exit, label %Loop
  br i1 %4, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %2
}

define i64 @add_neq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rax.0 = phi i64 [ 0, %Entry ], [ %2, %Loop ]
  %rcx.0 = phi i64 [ 0, %Entry ], [ %3, %Loop ]
  %1 = load i64, ptr %0, align 1
  %2 = add i64 %rax.0, %1
  %3 = add nuw nsw i64 %rcx.0, 1
  ; CHECK: %4 = icmp ne i64 %3, 5
  %4 = icmp ne i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %4, label %Exit, label %Loop
  br i1 %4, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %2
}

define i64 @sub_eq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rax.0 = phi i64 [ 0, %Entry ], [ %2, %Loop ]
  %rcx.0 = phi i64 [ 0, %Entry ], [ %3, %Loop ]
  %1 = load i64, ptr %0, align 1
  %2 = add i64 %rax.0, %1
  %3 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK: %4 = icmp eq i64 %3, 3
  %4 = icmp eq i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %4, label %Exit, label %Loop
  br i1 %4, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %2
}

define i64 @sub_neq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rax.0 = phi i64 [ 0, %Entry ], [ %2, %Loop ]
  %rcx.0 = phi i64 [ 0, %Entry ], [ %3, %Loop ]
  %1 = load i64, ptr %0, align 1
  %2 = add i64 %rax.0, %1
  %3 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK: %4 = icmp ne i64 %3, 3
  %4 = icmp ne i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %4, label %Exit, label %Loop
  br i1 %4, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %2
}
