;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt -peephole-opt-for-decompilation %s -S -o - | FileCheck %s

define i64 @add_eq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rcx.0 = phi i64 [ 0, %Entry ], [ %1, %Loop ]
  ; CHECK: %1 = add nuw nsw i64 %rcx.0, 1
  %1 = add nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: %2 = icmp eq i64 %1, 5
  %2 = icmp eq i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %2, label %Exit, label %Loop
  br i1 %2, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %1
}

define i64 @add_neq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rcx.0 = phi i64 [ 0, %Entry ], [ %1, %Loop ]
  ; CHECK: %1 = add nuw nsw i64 %rcx.0, 1
  %1 = add nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: %2 = icmp ne i64 %1, 5
  %2 = icmp ne i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %2, label %Exit, label %Loop
  br i1 %2, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %1
}

define i64 @sub_eq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rcx.0 = phi i64 [ 0, %Entry ], [ %1, %Loop ]
  ; CHECK: %1 = sub nuw nsw i64 %rcx.0, 1
  %1 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: %2 = icmp eq i64 %1, 3
  %2 = icmp eq i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %2, label %Exit, label %Loop
  br i1 %2, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %1
}

define i64 @sub_neq(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rcx.0 = phi i64 [ 0, %Entry ], [ %1, %Loop ]
  ; CHECK: %1 = sub nuw nsw i64 %rcx.0, 1
  %1 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: %2 = icmp ne i64 %1, 3
  %2 = icmp ne i64 %rcx.0, 4
  ; CHECK-NEXT: br i1 %2, label %Exit, label %Loop
  br i1 %2, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %1
}

define i64 @reorder(ptr %0) {
Entry:
  br label %Loop

Loop:                 ; preds = %Loop, %Entry
  %rcx.0 = phi i64 [ 0, %Entry ], [ %2, %Loop ]
  ; CHECK: %1 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: %2 = icmp ne i64 %1, 3
  %1 = icmp ne i64 %rcx.0, 4
  %2 = sub nuw nsw i64 %rcx.0, 1
  ; CHECK-NEXT: br i1 %2, label %Exit, label %Loop
  br i1 %1, label %Exit, label %Loop

Exit:                 ; preds = %Loop
  ret i64 %2
}
