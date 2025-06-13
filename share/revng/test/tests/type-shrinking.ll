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
