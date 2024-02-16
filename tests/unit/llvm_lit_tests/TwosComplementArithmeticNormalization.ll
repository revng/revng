;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -twoscomplement-normalization -S -o - | FileCheck %s
;
; Ensures that TwosComplementArithmeticNormalizationPass has transformed arithmetic operations appropriately.

define i8 @twoscomplement_norm_add_8_bitwidth (i8 %0) {
  ; CHECK: %2 = sub i8 %0, 1
  %2 = add i8 %0, 255
  ret i8 %2
}

define i8 @twoscomplement_norm_mul_8_bitwidth (i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 1)
  ; CHECK-NEXT: %3 = mul i8 %0, %2
  %2 = mul i8 %0, 255
  ret i8 %2
}

define i16 @twoscomplement_norm_add_16_bitwidth (i16 %0) {
  ; CHECK: %2 = sub i16 %0, 1
  %2 = add i16 %0, 65535
  ret i16 %2
}

define i32 @twoscomplement_norm_sub_32_bitwidth (i32 %0) {
  ; CHECK: %2 = add i32 %0, 1
  %2 = sub i32 %0, 4294967295
  ret i32 %2
}

define i64 @twoscomplement_norm_sub_64_bitwidth (i64 %0) {
  ; CHECK: %2 = add i64 %0, 1
  %2 = sub i64 %0, 18446744073709551615
  ret i64 %2
}

define i1 @twoscomplement_norm_unary_minus(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = icmp eq i32 %0, %2
  %2 = icmp eq i32 %0, 4294967295
  ret i1 %2
}

define i32 @twoscomplement_norm_binary_not(i32 %0) {
  ; CHECK: %2 = call i32 @binary_not(i32 %0)
  %2 = xor i32 %0, 4294967295
  ret i32 %2
}

define i1 @twoscomplement_norm_unary_minus_i16(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp eq i16 %0, %2
  %2 = icmp eq i16 %0, 65535
  ret i1 %2
}

define i1 @twoscomplement_norm_unary_minus_minus64(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 64)
  ; CHECK-NEXT: %3 = icmp eq i8 %0, %2
  %2 = icmp eq i8 %0, 192
  ret i1 %2
}

define i1 @twoscomplement_norm_unary_minus_minus127(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 127)
  ; CHECK-NEXT: %3 = icmp eq i8 %0, %2
  %2 = icmp eq i8 %0, 129
  ret i1 %2
}

define i32 @twoscomplement_norm_mul_unary_minus(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = mul i32 %0, %2
  %2 = mul i32 %0, 4294967295
  ret i32 %2
}

define i32 @twoscomplement_norm_mul_unary_minus_rev_arg_order(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = mul i32 %0, %2
  %2 = mul i32 4294967295, %0
  ret i32 %2
}

define i32 @twoscomplement_norm_sdiv_unary_minus(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = sdiv i32 %0, %2
  %2 = sdiv i32 %0, 4294967294
  ret i32 %2
}

define i32 @twoscomplement_norm_sdiv_unary_minus_2(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = sdiv i32 %2, %0
  %2 = sdiv i32 4294967294, %0
  ret i32 %2
}

define i32 @twoscomplement_norm_srem_unary_minus(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = srem i32 %0, %2
  %2 = srem i32 %0, 4294967294
  ret i32 %2
}

define i32 @twoscomplement_norm_srem_unary_minus_2(i32 %0) {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = srem i32 %2, %0
  %2 = srem i32 4294967294, %0
  ret i32 %2
}

define i1 @twoscomplement_norm_move_const_add_eq(i16 %0) {
  ; CHECK: %2 = icmp eq i16 %0, 1
  %2 = add i16 %0, 2
  %3 = icmp eq i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sub_eq(i16 %0) {
  ; CHECK: %2 = icmp eq i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp eq i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_add_ne(i16 %0) {
  ; CHECK: %2 = icmp ne i16 %0, 1
  %2 = add i16 %0, 2
  %3 = icmp ne i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sub_ne(i16 %0) {
  ; CHECK: %2 = icmp ne i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp ne i16 %2, 3
  ret i1 %3
}

define i1  @twoscomplement_norm_move_const_unary_minus(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 127)
  ; CHECK-NEXT: %3 = icmp eq i8 %0, %2
  %2 = add i8 %0, 1
  %3 = icmp eq i8 %2, 130
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_slt_ov(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 120)
  ; CHECK-NEXT: %3 = icmp slt i8 %0, %2
  %2 = sub i8 %0, 10
  %3 = icmp slt i8 %2, 126
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sle_ov(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 120)
  ; CHECK-NEXT: %3 = icmp sle i8 %0, %2
  %2 = sub i8 %0, 10
  %3 = icmp sle i8 %2, 126
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sgt_ov(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 120)
  ; CHECK-NEXT: %3 = icmp sgt i8 %0, %2
  %2 = sub i8 %0, 10
  %3 = icmp sgt i8 %2, 126
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sge_ov(i8 %0) {
  ; CHECK: %2 = call i8 @unary_minus(i8 120)
  ; CHECK-NEXT: %3 = icmp sge i8 %0, %2
  %2 = sub i8 %0, 10
  %3 = icmp sge i8 %2, 126
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_slt_unary_minus(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp slt i16 %0, %2
  %2 = sub i16 %0, 1
  %3 = icmp slt i16 %2, 65534
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sle_unary_minus(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp sle i16 %0, %2
  %2 = sub i16 %0, 1
  %3 = icmp sle i16 %2, 65534
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sgt_unary_minus(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp sgt i16 %0, %2
  %2 = sub i16 %0, 1
  %3 = icmp sgt i16 %2, 65534
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_sge_unary_minus(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp sge i16 %0, %2
  %2 = sub i16 %0, 1
  %3 = icmp sge i16 %2, 65534
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ult_ov(i16 %0) {
  ; CHECK: %2 = icmp ult i16 %0, 1
  %2 = sub i16 %0, 2
  %3 = icmp ult i16 %2, 65535
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ule_ov(i16 %0) {
  ; CHECK: %2 = icmp ule i16 %0, 1
  %2 = sub i16 %0, 2
  %3 = icmp ule i16 %2, 65535
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ugt_ov(i16 %0) {
  ; CHECK: %2 = icmp ugt i16 %0, 1
  %2 = sub i16 %0, 2
  %3 = icmp ugt i16 %2, 65535
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_uge_ov(i16 %0) {
  ; CHECK: %2 = icmp uge i16 %0, 1
  %2 = sub i16 %0, 2
  %3 = icmp uge i16 %2, 65535
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ult(i16 %0) {
  ; CHECK: %2 = icmp ult i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp ult i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ule(i16 %0) {
  ; CHECK: %2 = icmp ule i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp ule i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_ugt(i16 %0) {
  ; CHECK: %2 = icmp ugt i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp ugt i16 %2, 3
  ret i1 %3
}

define i1 @twoscomplement_norm_move_const_uge(i16 %0) {
  ; CHECK: %2 = icmp uge i16 %0, 5
  %2 = sub i16 %0, 2
  %3 = icmp uge i16 %2, 3
  ret i1 %3
}

define i1 @neg_neg(i16 %0) {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 7)
  ; CHECK-NEXT: %3 = icmp ult i16 %0, %2
  ; CHECK-NEXT: %4 = icmp uge i16 %0, 4
  ; CHECK-NEXT: %5 = and i1 %3, %4
  %2 = add i16 %0, -4
  %3 = icmp ult i16 %2, -11
  ret i1 %3
}

define i1 @twoscomplement_norm_boolean_not(i8 %0) {
  ; CHECK: %2 = call i1 @boolean_not(i8 %0)
  ; CHECK-NEXT: br i1 %2, label %If, label %Else

  %2 = icmp eq i8 %0, 0
  br i1 %2, label %If, label %Else
  If:
  ret i1 0
  Else:
  ret i1 1
}
