;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -S -early-type-shrinking -verify %s | FileCheck %s
; RUN: %root/bin/revng opt -S -early-type-shrinking -type-shrinking -instcombine -early-type-shrinking -type-shrinking -early-cse -dce -verify %s | FileCheck --check-prefix=PIPE %s

; CHECK exercises the pass alone, checking the value reaching each consumer.
; PIPE checks that narrowing is recovered after InstCombine canonicalizes it
; back into wide arithmetic.

; A plain `movslq`: the shift amounts are equal, so the cast is all that is
; left.
define i64 @movslq(i64 %0) {
  ; CHECK-LABEL: @movslq
  %shl = shl i64 %0, 32
  %ashr = ashr i64 %shl, 32
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %0 to i32
  ; CHECK: %[[E:[^ ]+]] = sext i32 %[[T]] to i64
  ; CHECK: ret i64 %[[E]]
  ret i64 %ashr
}

; A `movslq` fused with an array scale: the cast is shifted back up by what
; the two amounts leave over.
define i64 @movslq_scaled(i64 %0) {
  ; CHECK-LABEL: @movslq_scaled
  %shl = shl i64 %0, 32
  %ashr = ashr i64 %shl, 29
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %0 to i32
  ; CHECK: %[[E:[^ ]+]] = sext i32 %[[T]] to i64
  ; CHECK: %[[S:[^ ]+]] = shl i64 %[[E]], 3
  ; CHECK: ret i64 %[[S]]
  ret i64 %ashr
}

; The unsigned counterpart of the above, a `mov` of a 32-bit register.
define i64 @movl(i64 %0) {
  ; CHECK-LABEL: @movl
  %shl = shl i64 %0, 32
  %lshr = lshr i64 %shl, 32
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %0 to i32
  ; CHECK: %[[E:[^ ]+]] = zext i32 %[[T]] to i64
  ; CHECK: ret i64 %[[E]]
  ret i64 %lshr
}

; A 32-bit signed comparison between two 64-bit registers. There is no `shr`
; in this shape at all.
define i1 @compare32(i64 %0, i64 %1) {
  ; CHECK-LABEL: @compare32
  %a = shl i64 %0, 32
  %b = shl i64 %1, 32
  ; CHECK: trunc i64 %{{[^ ]+}} to i32
  ; CHECK: trunc i64 %{{[^ ]+}} to i32
  ; CHECK: %[[C:[^ ]+]] = icmp slt i32 %{{[^ ]+}}, %{{[^ ]+}}
  ; CHECK: ret i1 %[[C]]
  %cmp = icmp slt i64 %a, %b
  ret i1 %cmp
}

; Round the inner width up to i8, retaining the shifts instead of emitting i1
; arithmetic, which the model cannot represent.
define i64 @sign_broadcast(i64 %0) {
  ; CHECK-LABEL: @sign_broadcast
  %shl = shl i64 %0, 63
  %ashr = ashr i64 %shl, 63
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %0 to i8
  ; CHECK-NEXT: %[[L:[^ ]+]] = shl i8 %[[T]], 7
  ; CHECK-NEXT: %[[R:[^ ]+]] = ashr i8 %[[L]], 7
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[R]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %ashr
}

; Constants between the shifts participate in the same modular arithmetic.
define i64 @add_i8(i64 %x) {
  ; CHECK-LABEL: @add_i8(
  ; PIPE-LABEL: @add_i8(
  %a = shl i64 %x, 56
  %b = add i64 %a, 72057594037927936
  %r = ashr exact i64 %b, 56
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[A:[^ ]+]] = add i8 %[[T]], 1
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[A]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ; PIPE: add i8
  ; PIPE: sext i8
  ; PIPE: ret i64
  ret i64 %r
}

; A multiply supplies the factor even without an explicit left shift.
define i64 @mul_i16(i64 %x) {
  ; CHECK-LABEL: @mul_i16(
  ; PIPE-LABEL: @mul_i16(
  %p = mul i64 %x, 844424930131968
  %r = ashr exact i64 %p, 48
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i16
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i16 %[[T]], 3
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i16 %[[P]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ; PIPE: mul i16
  ; PIPE: sext i16
  ; PIPE: ret i64
  ret i64 %r
}

; Preserve subtraction order, including a constant on the left.
define i64 @sub_i32(i64 %x) {
  ; CHECK-LABEL: @sub_i32(
  %a = shl i64 %x, 32
  %b = sub i64 %a, 30064771072
  %r = ashr i64 %b, 32
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: %[[S:[^ ]+]] = sub i32 %[[T]], 7
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i32 %[[S]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

define i64 @sub_constant_left(i64 %x) {
  ; CHECK-LABEL: @sub_constant_left(
  %a = shl i64 %x, 32
  %b = sub i64 30064771072, %a
  %r = ashr i64 %b, 32
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: %[[S:[^ ]+]] = sub i32 7, %[[T]]
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i32 %[[S]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; General expression trees, not just chains with one variable operand.
define i64 @variable_tree(i64 %x, i64 %y, i64 %z) {
  ; CHECK-LABEL: @variable_tree(
  %a = shl i64 %x, 56
  %b = shl i64 %y, 56
  %c = shl i64 %z, 56
  %sum = add i64 %a, %b
  %bits = xor i64 %sum, %c
  %r = ashr i64 %bits, 56
  ; CHECK: %[[X:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[Y:[^ ]+]] = trunc i64 %y to i8
  ; CHECK-NEXT: %[[S:[^ ]+]] = add i8 %[[X]], %[[Y]]
  ; CHECK-NEXT: %[[Z:[^ ]+]] = trunc i64 %z to i8
  ; CHECK-NEXT: %[[B:[^ ]+]] = xor i8 %[[S]], %[[Z]]
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[B]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; Multiplication combines factors from both operands, even when neither can
; independently be narrowed to a supported width.
define i64 @variable_product(i64 %x, i64 %y) {
  ; CHECK-LABEL: @variable_product(
  %a = shl i64 %x, 28
  %b = shl i64 %y, 28
  %p = mul i64 %a, %b
  %r = ashr i64 %p, 56
  ; CHECK: %[[X:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[Y:[^ ]+]] = trunc i64 %y to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8 %[[X]], %[[Y]]
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[P]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; Both original operands share %p. Rebuild it once, preserving its wide user.
define i64 @shared_expression(i64 %x, ptr %out) {
  ; CHECK-LABEL: @shared_expression(
  %p = mul i64 %x, 360287970189639680
  store i64 %p, ptr %out
  %bits = and i64 %p, -72057594037927936
  %sum = add i64 %bits, %p
  %r = ashr i64 %sum, 56
  ; CHECK: store i64 %p, ptr %out
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8 %[[T]], 5
  ; CHECK-NEXT: %[[B:[^ ]+]] = and i8 %[[P]], {{.*}}
  ; CHECK-NEXT: %[[S:[^ ]+]] = add i8 %[[B]], %[[P]]
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[S]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; All bitwise operations use the same factoring rule.
define i64 @constant_chain(i64 %x) {
  ; CHECK-LABEL: @constant_chain(
  ; PIPE-LABEL: @constant_chain(
  %p = mul i64 %x, 360287970189639680
  %a = xor i64 %p, -8863084066665136128
  %b = or i64 %a, 72057594037927936
  %c = and i64 %b, -72057594037927936
  %d = add i64 %c, 72057594037927936
  %r = ashr exact i64 %d, 56
  ; CHECK: trunc i64 %x to i8
  ; CHECK-NEXT: mul i8
  ; CHECK-NEXT: xor i8
  ; CHECK-NEXT: or i8
  ; CHECK-NEXT: and i8
  ; CHECK-NEXT: %[[A:[^ ]+]] = add i8
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[A]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ; PIPE: mul i8
  ; PIPE: xor i8
  ; PIPE: add i8
  ; PIPE: sext i8
  ret i64 %r
}

; 45 trailing zeros permit i32, with a remaining inner arithmetic shift.
define i64 @rounded_multiply(i64 %x) {
  ; CHECK-LABEL: @rounded_multiply(
  %p = mul i64 %x, 105553116266496
  %r = ashr i64 %p, 56
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i32 %[[T]], u0x6000
  ; CHECK-NEXT: %[[S:[^ ]+]] = ashr i32 %[[P]], 24
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i32 %[[S]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

define i64 @outer_shift(i64 %x) {
  ; CHECK-LABEL: @outer_shift(
  %p = mul i64 %x, 844424930131968
  %r = lshr i64 %p, 40
  ; CHECK: trunc i64 %x to i16
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i16
  ; CHECK-NEXT: %[[E:[^ ]+]] = zext i16 %[[P]] to i64
  ; CHECK-NEXT: %[[S:[^ ]+]] = shl i64 %[[E]], 8
  ; CHECK-NEXT: ret i64 %[[S]]
  ret i64 %r
}

; Nested shifts are combined, and flags are not transferred to new arithmetic.
define i64 @nested_flagged_shifts(i64 %x) {
  ; CHECK-LABEL: @nested_flagged_shifts(
  %a = shl nuw nsw i64 %x, 24
  %b = shl nuw nsw i64 %a, 32
  %c = add nuw nsw i64 %b, 72057594037927936
  %r = ashr exact i64 %c, 56
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[A:[^ ]+]] = add i8 %[[T]], 1
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[A]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; The residual shift on the second multiplicand would exceed i8's width.
; Its contribution is zero, not poison.
define i64 @discarded_product(i64 %x, i64 %y) {
  ; CHECK-LABEL: @discarded_product(
  %a = shl i64 %x, 8
  %b = shl i64 %y, 63
  %p = mul i64 %a, %b
  %r = ashr i64 %p, 56
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8 %[[T]], 0
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[P]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ; PIPE-LABEL: @discarded_product(
  ; PIPE: ret i64 0
  ret i64 %r
}

; Comparisons use the common factor even with different explicit shifts.
define i1 @unequal_shift_compare(i64 %x, i64 %y) {
  ; CHECK-LABEL: @unequal_shift_compare(
  %a = shl i64 %x, 56
  %b = shl i64 %y, 57
  %r = icmp slt i64 %a, %b
  ; CHECK: %[[X:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[Y:[^ ]+]] = trunc i64 %y to i8
  ; CHECK-NEXT: %[[S:[^ ]+]] = shl i8 %[[Y]], 1
  ; CHECK-NEXT: %[[C:[^ ]+]] = icmp slt i8 %[[X]], %[[S]]
  ; CHECK-NEXT: ret i1 %[[C]]
  ret i1 %r
}

define i1 @constant_compare(i64 %x) {
  ; CHECK-LABEL: @constant_compare(
  %p = mul i64 %x, 360287970189639680
  %r = icmp ne i64 0, %p
  ; CHECK: trunc i64 %x to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8
  ; CHECK-NEXT: %[[C:[^ ]+]] = icmp ne i8 0, %[[P]]
  ; CHECK-NEXT: ret i1 %[[C]]
  ret i1 %r
}

; A low nonzero subtrahend borrows into the retained bits.
define i64 @unaligned_subtract(i64 %x) {
  ; CHECK-LABEL: @unaligned_subtract(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, 56
  %b = sub i64 %a, 1
  %r = ashr i64 %b, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

; Bit 44 survives the right shift. It prevents an i16 rewrite, but an i32
; rewrite still retains it, demonstrating conservative width selection.
define i64 @retained_constant_bit(i64 %x) {
  ; CHECK-LABEL: @retained_constant_bit(
  %a = shl i64 %x, 48
  %b = or i64 %a, 17592186044416
  %r = lshr i64 %b, 40
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: %[[L:[^ ]+]] = shl i32 %[[T]], 16
  ; CHECK-NEXT: %[[O:[^ ]+]] = or i32 %[[L]], u0x1000
  ; CHECK-NEXT: %[[R:[^ ]+]] = lshr i32 %[[O]], 8
  ; CHECK-NEXT: %[[E:[^ ]+]] = zext i32 %[[R]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

define i64 @insufficient_factor(i64 %x) {
  ; CHECK-LABEL: @insufficient_factor(
  ; CHECK-NOT: trunc
  %p = mul i64 %x, 6
  %r = ashr i64 %p, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

define i64 @variable_left_shift(i64 %x, i64 %amount) {
  ; CHECK-LABEL: @variable_left_shift(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, %amount
  %r = ashr i64 %a, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

define i64 @variable_right_shift(i64 %x, i64 %amount) {
  ; CHECK-LABEL: @variable_right_shift(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, 56
  %r = ashr i64 %a, %amount
  ; CHECK: ret i64 %r
  ret i64 %r
}

define i64 @invalid_left_shift(i64 %x) {
  ; CHECK-LABEL: @invalid_left_shift(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, 64
  %r = ashr i64 %a, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

define i64 @invalid_right_shift(i64 %x) {
  ; CHECK-LABEL: @invalid_right_shift(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, 56
  %r = ashr i64 %a, 64
  ; CHECK: ret i64 %r
  ret i64 %r
}

; freeze can destroy known alignment when its operand is poison.
define i64 @frozen_shift(i64 %x) {
  ; CHECK-LABEL: @frozen_shift(
  ; CHECK-NOT: trunc
  %a = shl i64 %x, 56
  %f = freeze i64 %a
  %r = ashr i64 %f, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

; A multiply can supply the whole factor from one operand, so an opaque second
; operand does not block the rewrite: it is rebuilt by truncating it, and the
; wide value stays live. This is the only shape that narrows while keeping one
; of its original operands.
define i64 @partial_rewrite(i64 %x, i64 %y) {
  ; CHECK-LABEL: @partial_rewrite(
  %o = freeze i64 %y
  %s = shl i64 %x, 56
  %p = mul i64 %s, %o
  %r = ashr i64 %p, 56
  ; CHECK: %[[T:[^ ]+]] = trunc i64 %x to i8
  ; CHECK-NEXT: %[[O:[^ ]+]] = trunc i64 %o to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8 %[[T]], %[[O]]
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[P]] to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

define <2 x i64> @vector_shift(<2 x i64> %x) {
  ; CHECK-LABEL: @vector_shift(
  ; CHECK-NOT: trunc
  %a = shl <2 x i64> %x, <i64 56, i64 56>
  %r = ashr <2 x i64> %a, <i64 56, i64 56>
  ; CHECK: ret <2 x i64> %r
  ret <2 x i64> %r
}

; The rules depend on the operand width, not on a 64-bit source type.
define i16 @small_integer(i16 %x) {
  ; CHECK-LABEL: @small_integer(
  %a = shl i16 %x, 8
  %b = add i16 %a, 256
  %r = ashr i16 %b, 8
  ; CHECK: %[[T:[^ ]+]] = trunc i16 %x to i8
  ; CHECK-NEXT: %[[A:[^ ]+]] = add i8 %[[T]], 1
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[A]] to i16
  ; CHECK-NEXT: ret i16 %[[E]]
  ret i16 %r
}

; Constants need APInt arithmetic, including those wider than uint64_t.
define i128 @wide_integer(i128 %x) {
  ; CHECK-LABEL: @wide_integer(
  %p = mul i128 %x, 6646139978924579364519035301401722880
  %r = ashr i128 %p, 120
  ; CHECK: %[[T:[^ ]+]] = trunc i128 %x to i8
  ; CHECK-NEXT: %[[P:[^ ]+]] = mul i8 %[[T]], 5
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 %[[P]] to i128
  ; CHECK-NEXT: ret i128 %[[E]]
  ret i128 %r
}

; A chain longer than the traversal budget leaves the operands it never reaches
; opaque, just as the freeze in partial_rewrite does. Here that is the whole
; input, so its exponent collapses to zero and the shift is left alone.
define i64 @bounded_chain(i64 %x) {
  ; CHECK-LABEL: @bounded_chain(
  ; CHECK-NOT: trunc
  %v0 = shl i64 %x, 56
  %v1 = add i64 %v0, 72057594037927936
  %v2 = add i64 %v1, 72057594037927936
  %v3 = add i64 %v2, 72057594037927936
  %v4 = add i64 %v3, 72057594037927936
  %v5 = add i64 %v4, 72057594037927936
  %v6 = add i64 %v5, 72057594037927936
  %v7 = add i64 %v6, 72057594037927936
  %v8 = add i64 %v7, 72057594037927936
  %v9 = add i64 %v8, 72057594037927936
  %v10 = add i64 %v9, 72057594037927936
  %v11 = add i64 %v10, 72057594037927936
  %v12 = add i64 %v11, 72057594037927936
  %v13 = add i64 %v12, 72057594037927936
  %v14 = add i64 %v13, 72057594037927936
  %v15 = add i64 %v14, 72057594037927936
  %v16 = add i64 %v15, 72057594037927936
  %v17 = add i64 %v16, 72057594037927936
  %v18 = add i64 %v17, 72057594037927936
  %v19 = add i64 %v18, 72057594037927936
  %v20 = add i64 %v19, 72057594037927936
  %v21 = add i64 %v20, 72057594037927936
  %v22 = add i64 %v21, 72057594037927936
  %v23 = add i64 %v22, 72057594037927936
  %v24 = add i64 %v23, 72057594037927936
  %v25 = add i64 %v24, 72057594037927936
  %v26 = add i64 %v25, 72057594037927936
  %v27 = add i64 %v26, 72057594037927936
  %v28 = add i64 %v27, 72057594037927936
  %v29 = add i64 %v28, 72057594037927936
  %v30 = add i64 %v29, 72057594037927936
  %v31 = add i64 %v30, 72057594037927936
  %v32 = add i64 %v31, 72057594037927936
  %v33 = add i64 %v32, 72057594037927936
  %v34 = add i64 %v33, 72057594037927936
  %v35 = add i64 %v34, 72057594037927936
  %v36 = add i64 %v35, 72057594037927936
  %v37 = add i64 %v36, 72057594037927936
  %v38 = add i64 %v37, 72057594037927936
  %v39 = add i64 %v38, 72057594037927936
  %v40 = add i64 %v39, 72057594037927936
  %v41 = add i64 %v40, 72057594037927936
  %v42 = add i64 %v41, 72057594037927936
  %v43 = add i64 %v42, 72057594037927936
  %v44 = add i64 %v43, 72057594037927936
  %v45 = add i64 %v44, 72057594037927936
  %v46 = add i64 %v45, 72057594037927936
  %v47 = add i64 %v46, 72057594037927936
  %v48 = add i64 %v47, 72057594037927936
  %v49 = add i64 %v48, 72057594037927936
  %v50 = add i64 %v49, 72057594037927936
  %v51 = add i64 %v50, 72057594037927936
  %v52 = add i64 %v51, 72057594037927936
  %v53 = add i64 %v52, 72057594037927936
  %v54 = add i64 %v53, 72057594037927936
  %v55 = add i64 %v54, 72057594037927936
  %v56 = add i64 %v55, 72057594037927936
  %v57 = add i64 %v56, 72057594037927936
  %v58 = add i64 %v57, 72057594037927936
  %v59 = add i64 %v58, 72057594037927936
  %v60 = add i64 %v59, 72057594037927936
  %v61 = add i64 %v60, 72057594037927936
  %v62 = add i64 %v61, 72057594037927936
  %v63 = add i64 %v62, 72057594037927936
  %v64 = add i64 %v63, 72057594037927936
  %r = ashr i64 %v64, 56
  ; CHECK: ret i64 %r
  ret i64 %r
}

; Count distinct instructions, not paths through a shared expression DAG.
define i64 @shared_dag(i64 %x) {
  ; CHECK-LABEL: @shared_dag(
  %v0 = shl i64 %x, 56
  %v1 = add i64 %v0, %v0
  %v2 = add i64 %v1, %v1
  %v3 = add i64 %v2, %v2
  %v4 = add i64 %v3, %v3
  %v5 = add i64 %v4, %v4
  %v6 = add i64 %v5, %v5
  %v7 = add i64 %v6, %v6
  %v8 = add i64 %v7, %v7
  %v9 = add i64 %v8, %v8
  %v10 = add i64 %v9, %v9
  %v11 = add i64 %v10, %v10
  %v12 = add i64 %v11, %v11
  %v13 = add i64 %v12, %v12
  %v14 = add i64 %v13, %v13
  %v15 = add i64 %v14, %v14
  %v16 = add i64 %v15, %v15
  %v17 = add i64 %v16, %v16
  %v18 = add i64 %v17, %v17
  %v19 = add i64 %v18, %v18
  %v20 = add i64 %v19, %v19
  %r = ashr i64 %v20, 56
  ; CHECK: trunc i64 %x to i8
  ; CHECK-COUNT-20: add i8
  ; CHECK-NEXT: %[[E:[^ ]+]] = sext i8 {{.*}} to i64
  ; CHECK-NEXT: ret i64 %[[E]]
  ret i64 %r
}

; Each operand gets its own traversal budget, so a long operand cannot starve
; the other. Together these two chains exceed the budget, so sharing one budget
; between them would leave the second operand opaque and make the common
; exponent zero.
define i1 @comparison_budget(i64 %x, i64 %y) {
  ; CHECK-LABEL: @comparison_budget(
  %a0 = shl i64 %x, 56
  %a1 = add i64 %a0, 72057594037927936
  %a2 = add i64 %a1, 72057594037927936
  %a3 = add i64 %a2, 72057594037927936
  %a4 = add i64 %a3, 72057594037927936
  %a5 = add i64 %a4, 72057594037927936
  %a6 = add i64 %a5, 72057594037927936
  %a7 = add i64 %a6, 72057594037927936
  %a8 = add i64 %a7, 72057594037927936
  %a9 = add i64 %a8, 72057594037927936
  %a10 = add i64 %a9, 72057594037927936
  %a11 = add i64 %a10, 72057594037927936
  %a12 = add i64 %a11, 72057594037927936
  %a13 = add i64 %a12, 72057594037927936
  %a14 = add i64 %a13, 72057594037927936
  %a15 = add i64 %a14, 72057594037927936
  %a16 = add i64 %a15, 72057594037927936
  %a17 = add i64 %a16, 72057594037927936
  %a18 = add i64 %a17, 72057594037927936
  %a19 = add i64 %a18, 72057594037927936
  %a20 = add i64 %a19, 72057594037927936
  %a21 = add i64 %a20, 72057594037927936
  %a22 = add i64 %a21, 72057594037927936
  %a23 = add i64 %a22, 72057594037927936
  %a24 = add i64 %a23, 72057594037927936
  %a25 = add i64 %a24, 72057594037927936
  %a26 = add i64 %a25, 72057594037927936
  %a27 = add i64 %a26, 72057594037927936
  %a28 = add i64 %a27, 72057594037927936
  %a29 = add i64 %a28, 72057594037927936
  %a30 = add i64 %a29, 72057594037927936
  %a31 = add i64 %a30, 72057594037927936
  %a32 = add i64 %a31, 72057594037927936
  %b0 = shl i64 %y, 56
  %b1 = add i64 %b0, 72057594037927936
  %b2 = add i64 %b1, 72057594037927936
  %b3 = add i64 %b2, 72057594037927936
  %b4 = add i64 %b3, 72057594037927936
  %b5 = add i64 %b4, 72057594037927936
  %b6 = add i64 %b5, 72057594037927936
  %b7 = add i64 %b6, 72057594037927936
  %b8 = add i64 %b7, 72057594037927936
  %b9 = add i64 %b8, 72057594037927936
  %b10 = add i64 %b9, 72057594037927936
  %b11 = add i64 %b10, 72057594037927936
  %b12 = add i64 %b11, 72057594037927936
  %b13 = add i64 %b12, 72057594037927936
  %b14 = add i64 %b13, 72057594037927936
  %b15 = add i64 %b14, 72057594037927936
  %b16 = add i64 %b15, 72057594037927936
  %b17 = add i64 %b16, 72057594037927936
  %b18 = add i64 %b17, 72057594037927936
  %b19 = add i64 %b18, 72057594037927936
  %b20 = add i64 %b19, 72057594037927936
  %b21 = add i64 %b20, 72057594037927936
  %b22 = add i64 %b21, 72057594037927936
  %b23 = add i64 %b22, 72057594037927936
  %b24 = add i64 %b23, 72057594037927936
  %b25 = add i64 %b24, 72057594037927936
  %b26 = add i64 %b25, 72057594037927936
  %b27 = add i64 %b26, 72057594037927936
  %b28 = add i64 %b27, 72057594037927936
  %b29 = add i64 %b28, 72057594037927936
  %b30 = add i64 %b29, 72057594037927936
  %b31 = add i64 %b30, 72057594037927936
  %b32 = add i64 %b31, 72057594037927936
  %cmp = icmp slt i64 %a32, %b32
  ; CHECK: %[[X:[^ ]+]] = trunc i64 %x to i8
  ; CHECK: %[[Y:[^ ]+]] = trunc i64 %y to i8
  ; CHECK: %[[C:[^ ]+]] = icmp slt i8 %{{[^ ]+}}, %{{[^ ]+}}
  ; CHECK-NEXT: ret i1 %[[C]]
  ret i1 %cmp
}
