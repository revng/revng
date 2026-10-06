;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -simplify-comparisons -verify %s -S -o - | FileCheck %s

; Bit 7 of `%shifted` is bit 31 of `%x`, hence the sign bit of its low i32.
;
;   %shifted = lshr i64 %x, 24       %value = trunc i64 %x to i32
;   %sign = and i64 %shifted, 128    %result = icmp slt i32 %value, 0
;   %result = icmp ne i64 %sign, 0
;
; CHECK-LABEL: define i1 @shifted_sign_bit(i64 %x)
define i1 @shifted_sign_bit(i64 %x) {
entry:
  %shifted = lshr i64 %x, 24
  %sign = and i64 %shifted, 128
  %result = icmp ne i64 %sign, 0
  ; CHECK: [[VALUE:%[a-zA-Z0-9._]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = icmp slt i32 [[VALUE]], 0
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %result
}

; Shifted masks are canonicalized even when they do not represent a sign bit.
;
; CHECK-LABEL: define i1 @shifted_mask(i64 %x)
define i1 @shifted_mask(i64 %x) {
entry:
  %shifted.once = lshr i64 %x, 8
  %shifted = ashr i64 %shifted.once, 16
  %masked = and i64 %shifted, 3
  %result = icmp ne i64 %masked, 0
  ; CHECK: [[MASKED:%[a-zA-Z0-9._]+]] = and i64 %x, u0x3000000
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = icmp ne i64 [[MASKED]], 0
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %result
}

; Mask bits observing sign extension cannot be moved using the simple identity.
;
; CHECK-LABEL: define i1 @sign_extended_mask(i32 %x)
define i1 @sign_extended_mask(i32 %x) {
entry:
  %shifted = ashr i32 %x, 8
  %masked = and i32 %shifted, 1073741824
  %result = icmp ne i32 %masked, 0
  ; CHECK: %shifted = ashr i32 %x, 8
  ; CHECK-NEXT: %masked = and i32 %shifted, u0x40000000
  ; CHECK-NEXT: %result = icmp ne i32 %masked, 0
  ; CHECK-NEXT: ret i1 %result
  ret i1 %result
}

; Building the replacement can fold both the truncation and the comparison.
; CHECK-LABEL: define i1 @constant_sign_bit_clear()
define i1 @constant_sign_bit_clear() {
  %mask = and i64 42, 128
  %result = icmp ne i64 %mask, 0
  ; CHECK: ret i1 false
  ret i1 %result
}

; The replacement can also fold when no truncation is needed.
; CHECK-LABEL: define i1 @constant_sign_bit_equal()
define i1 @constant_sign_bit_equal() {
  %mask = and i8 -1, 128
  %result = icmp eq i8 %mask, 0
  ; CHECK: ret i1 false
  ret i1 %result
}

; Consuming each OR allows distribution through the whole unshared tree.
; CHECK-LABEL: define i1 @or_tree_is_zero(
define i1 @or_tree_is_zero(i64 %a, i64 %b, i64 %c) {
  %ab = or i64 %a, %b
  %abc = or i64 %ab, %c
  %result = icmp eq i64 %abc, 0
  ; CHECK-NOT: or i64
  ; CHECK: [[AZERO:%[a-zA-Z0-9._]+]] = icmp eq i64 %a, 0
  ; CHECK-NEXT: [[BZERO:%[a-zA-Z0-9._]+]] = icmp eq i64 %b, 0
  ; CHECK-NEXT: [[ABZERO:%[a-zA-Z0-9._]+]] = and i1 [[AZERO]], [[BZERO]]
  ; CHECK-NEXT: [[CZERO:%[a-zA-Z0-9._]+]] = icmp eq i64 %c, 0
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = and i1 [[ABZERO]], [[CZERO]]
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %result
}

; Repeated operands form a compact DAG, not a tree to expand exponentially.
; Only the outermost, single-use OR is distributed, leaving two comparisons.
; CHECK-LABEL: define i1 @shared_or_chain(
define i1 @shared_or_chain(i64 %a, i64 %b) {
  %v0 = or i64 %a, %b
  %v1 = or i64 %v0, %v0
  %v2 = or i64 %v1, %v1
  %v3 = or i64 %v2, %v2
  %v4 = or i64 %v3, %v3
  %v5 = or i64 %v4, %v4
  %v6 = or i64 %v5, %v5
  %v7 = or i64 %v6, %v6
  %v8 = or i64 %v7, %v7
  %v9 = or i64 %v8, %v8
  %v10 = or i64 %v9, %v9
  %v11 = or i64 %v10, %v10
  %v12 = or i64 %v11, %v11
  %v13 = or i64 %v12, %v12
  %v14 = or i64 %v13, %v13
  %v15 = or i64 %v14, %v14
  %v16 = or i64 %v15, %v15
  %v17 = or i64 %v16, %v16
  %result = icmp ne i64 %v17, 0
  ; CHECK-COUNT-17: = or i64
  ; CHECK-NEXT: [[LEFT:%[a-zA-Z0-9._]+]] = icmp ne i64 %v16, 0
  ; CHECK-NEXT: [[RIGHT:%[a-zA-Z0-9._]+]] = icmp ne i64 %v16, 0
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = or i1 [[LEFT]], [[RIGHT]]
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %result
}

; Sharing through different operands must also stop distribution at the join.
; CHECK-LABEL: define i1 @shared_or_diamond(
define i1 @shared_or_diamond(i64 %a, i64 %b, i64 %c, i64 %d) {
  %shared = or i64 %a, %b
  %left = or i64 %shared, %c
  %right = or i64 %shared, %d
  %both = or i64 %left, %right
  %result = icmp eq i64 %both, 0
  ; CHECK: %shared = or i64 %a, %b
  ; CHECK-NEXT: [[LEFT:%[a-zA-Z0-9._]+]] = icmp eq i64 %shared, 0
  ; CHECK-NEXT: [[CZERO:%[a-zA-Z0-9._]+]] = icmp eq i64 %c, 0
  ; CHECK-NEXT: [[LZERO:%[a-zA-Z0-9._]+]] = and i1 [[LEFT]], [[CZERO]]
  ; CHECK-NEXT: [[RIGHT:%[a-zA-Z0-9._]+]] = icmp eq i64 %shared, 0
  ; CHECK-NEXT: [[DZERO:%[a-zA-Z0-9._]+]] = icmp eq i64 %d, 0
  ; CHECK-NEXT: [[RZERO:%[a-zA-Z0-9._]+]] = and i1 [[RIGHT]], [[DZERO]]
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = and i1 [[LZERO]], [[RZERO]]
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %result
}
