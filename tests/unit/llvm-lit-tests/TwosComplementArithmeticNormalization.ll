;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -twoscomplement-normalization -S -o - | FileCheck %s
;
; Ensures that TwosComplementArithmeticNormalizationPass has transformed arithmetic operations appropriately.

define i8 @twoscomplement_norm_add_8_bitwidth (i8 %0) !revng.tags !0 {
  ; CHECK: %2 = sub i8 %0, 1
  %2 = add i8 %0, 255
  ret i8 %2
}

define i8 @twoscomplement_norm_mul_8_bitwidth (i8 %0) !revng.tags !0 {
  ; CHECK: %2 = call i8 @unary_minus(i8 1)
  ; CHECK-NEXT: %3 = mul i8 %0, %2
  %2 = mul i8 %0, 255
  ret i8 %2
}

define i16 @twoscomplement_norm_add_16_bitwidth (i16 %0) !revng.tags !0 {
  ; CHECK: %2 = sub i16 %0, 1
  %2 = add i16 %0, 65535
  ret i16 %2
}

define i32 @twoscomplement_norm_sub_32_bitwidth (i32 %0) !revng.tags !0 {
  ; CHECK: %2 = add i32 %0, 1
  %2 = sub i32 %0, 4294967295
  ret i32 %2
}

define i64 @twoscomplement_norm_sub_64_bitwidth (i64 %0) !revng.tags !0 {
  ; CHECK: %2 = add i64 %0, 1
  %2 = sub i64 %0, 18446744073709551615
  ret i64 %2
}

define i1 @twoscomplement_norm_unary_minus(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = icmp eq i32 %0, %2
  %2 = icmp eq i32 %0, 4294967295
  ret i1 %2
}

define i32 @twoscomplement_norm_binary_not(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @binary_not(i32 %0)
  %2 = xor i32 %0, 4294967295
  ret i32 %2
}

define i1 @twoscomplement_norm_unary_minus_i16(i16 %0) !revng.tags !0 {
  ; CHECK: %2 = call i16 @unary_minus.2(i16 1)
  ; CHECK-NEXT: %3 = icmp eq i16 %0, %2
  %2 = icmp eq i16 %0, 65535
  ret i1 %2
}

define i1 @twoscomplement_norm_unary_minus_minus64(i8 %0) !revng.tags !0 {
  ; CHECK: %2 = call i8 @unary_minus(i8 64)
  ; CHECK-NEXT: %3 = icmp eq i8 %0, %2
  %2 = icmp eq i8 %0, 192
  ret i1 %2
}

define i1 @twoscomplement_norm_unary_minus_minus127(i8 %0) !revng.tags !0 {
  ; CHECK: %2 = call i8 @unary_minus(i8 127)
  ; CHECK-NEXT: %3 = icmp eq i8 %0, %2
  %2 = icmp eq i8 %0, 129
  ret i1 %2
}

define i32 @twoscomplement_norm_mul_unary_minus(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = mul i32 %0, %2
  %2 = mul i32 %0, 4294967295
  ret i32 %2
}

define i32 @twoscomplement_norm_mul_unary_minus_rev_arg_order(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 1)
  ; CHECK-NEXT: %3 = mul i32 %0, %2
  %2 = mul i32 4294967295, %0
  ret i32 %2
}

define i32 @twoscomplement_norm_sdiv_unary_minus(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = sdiv i32 %0, %2
  %2 = sdiv i32 %0, 4294967294
  ret i32 %2
}

define i32 @twoscomplement_norm_sdiv_unary_minus_2(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = sdiv i32 %2, %0
  %2 = sdiv i32 4294967294, %0
  ret i32 %2
}

define i32 @twoscomplement_norm_srem_unary_minus(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = srem i32 %0, %2
  %2 = srem i32 %0, 4294967294
  ret i32 %2
}

define i32 @twoscomplement_norm_srem_unary_minus_2(i32 %0) !revng.tags !0 {
  ; CHECK: %2 = call i32 @unary_minus.1(i32 2)
  ; CHECK-NEXT: %3 = srem i32 %2, %0
  %2 = srem i32 4294967294, %0
  ret i32 %2
}


!0 = !{!"Isolated"}
