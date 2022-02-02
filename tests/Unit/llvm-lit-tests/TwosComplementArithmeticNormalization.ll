; RUN: %revngopt %s -twoscomplement-normalization -S -o - | FileCheck %s
;
; Ensures that TwosComplementArithmeticNormalizationPass has transformed arithmetic operations appropriately.

define i8 @twoscomplement_norm_add_8_bitwidth (i8 %0) !revng.tags !0 {
  ; CHECK: %2 = sub i8 %0, 1
  %2 = add i8 %0, 255
  ret i8 %2
}

define i8 @twoscomplement_norm_mul_8_bitwidth (i8 %0) !revng.tags !0 {
  ; CHECK: %2 = mul i8 %0, -1
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

!0 = !{!"Lifted"}
