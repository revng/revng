;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; The pass replaces the uses of the shift pair and leaves the pair itself
; behind, so each case is checked through the value that reaches the `ret`.

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

; The usual sign broadcast. Its inner type would be `i1`, which no model
; primitive can be emitted as, so it is left alone.
define i64 @sign_broadcast(i64 %0) {
  ; CHECK-LABEL: @sign_broadcast
  ; CHECK-NOT: trunc
  %shl = shl i64 %0, 63
  %ashr = ashr i64 %shl, 63
  ; CHECK: ret i64 %ashr
  ret i64 %ashr
}
