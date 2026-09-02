;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

define i64 @sum32(i64 %0, i64 %1) {
  %3 = add i64 %1, %0
  ; CHECK: add i32
  ; CHECK-NOT: and
  %4 = and i64 %3, 4294967295
  ret i64 %4
}

define i64 @shl32(i64 %0) {
  %shl = shl i64 %0, 7
  ; CHECK: shl i32
  ; CHECK-NOT: and
  %masked = and i64 %shl, 4294967295
  ret i64 %masked
}

define i64 @ashr32(i64 %0) {
  %shl = shl i64 %0, 32
  ; CHECK: ashr i32 %{{.*}}, 1
  %ashr = ashr i64 %shl, 33
  %masked = and i64 %ashr, 4294967295
  ret i64 %masked
}

define i64 @lshr32(i64 %0) {
  %shl = shl i64 %0, 32
  ; CHECK: lshr i32 %{{.*}}, 1
  %ashr = lshr i64 %shl, 33
  %masked = and i64 %ashr, 4294967295
  ret i64 %masked
}

; A phi copies one of its incoming values bit for bit, so only the bits alive
; in the result are alive in each of them.
define i64 @phi32(i64 %0, i64 %1, i1 %c) {
entry:
  br i1 %c, label %then, label %else

then:
  %a = add i64 %0, %1
  br label %join

else:
  %b = sub i64 %0, %1
  br label %join

join:
  ; CHECK: phi i32
  %p = phi i64 [ %a, %then ], [ %b, %else ]
  ; CHECK-NOT: and
  %masked = and i64 %p, 4294967295
  ret i64 %masked
}

; A loop counter feeds nothing but the phi it comes back to, so before this
; the whole chain stayed 64 bits wide.
define i64 @loop_counter32(i64 %n) {
entry:
  br label %loop

loop:
  ; CHECK: phi i32
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  ; CHECK: add i32
  %next = add i64 %i, 1
  %masked_next = and i64 %next, 4294967295
  %masked_n = and i64 %n, 4294967295
  %cond = icmp ult i64 %masked_next, %masked_n
  br i1 %cond, label %loop, label %exit

exit:
  %r = and i64 %i, 4294967295
  ret i64 %r
}

; A select copies one of its two values bit for bit, the same as a phi. The
; condition picks between them rather than being one of them, so it stays as
; it is.
define i64 @select32(i64 %0, i64 %1, i1 %c) {
  %a = add i64 %0, %1
  %b = sub i64 %0, %1
  ; CHECK: select i1 %c, i32
  %s = select i1 %c, i64 %a, i64 %b
  ; CHECK-NOT: and
  %masked = and i64 %s, 4294967295
  ret i64 %masked
}
