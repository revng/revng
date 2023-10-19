;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -pretty-int-formatting -S -o - | FileCheck %s

define i8 @check_char(i8 %x) !revng.tags !0 {
  ; CHECK: %1 = call i8 @print_char(i8 97)
  ; CHECK-NEXT: %y = add i8 %x, %1
  %y = add i8 %x, 97
  ; CHECK: %2 = call i8 @print_char(i8 1)
  ; CHECK-NEXT: %z = add i8 %x, %2
  %z = add i8 %x, 1
  ret i8 %z
}

define i32 @check_hex(i32 %x) !revng.tags !0 {
  ; CHECK: %1 = call i32 @print_hex(i32 255)
  ; CHECK-NEXT: %z = shl i32 %1, %x
  %z = shl i32 255, %x
  %y = shl i32 %x, 255
  ; CHECK: %2 = call i32 @print_hex(i32 15)
  ; CHECK-NEXT: %k = lshr i32 %2, 2
  %k = lshr i32 15, 2
  ; CHECK: %3 = call i32 @print_hex(i32 15)
  ; CHECK-NEXT: %m = ashr i32 %3, 2
  %m = ashr i32 15, 2
  ret i32 %z
}

define void @check_hex_and_or_xor(i32 %x) !revng.tags !0 {
  ; CHECK: %1 = call i32 @print_hex(i32 255)
  ; CHECK-NEXT: %z = or i32 %1, %x
  %z = or i32 255, %x
  ; CHECK: %2 = call i32 @print_hex(i32 1)
  ; CHECK-NEXT: %y = or i32 %x, %2
  %y = or i32 %x, 1
  ; CHECK: %3 = call i32 @print_hex(i32 1)
  ; CHECK-NEXT: %l = xor i32 %x, %3
  %l = xor i32 %x, 1
  ; CHECK: %4 = call i32 @print_hex(i32 1)
  ; CHECK-NEXT: %k = xor i32 %4, %x
  %k = xor i32 1, %x
  ; CHECK: %5 = call i32 @print_hex(i32 1)
  ; CHECK-NEXT: %m = and i32 %x, %5
  %m = and i32 %x, 1
  ; CHECK: %6 = call i32 @print_hex(i32 1)
  ; CHECK-NEXT: %n = and i32 %6, %x
  %n = and i32 1, %x
  ret void
}

define i1 @check_bool(i1 %print) !revng.tags !0 {
  ; CHECK: %1 = call i1 @print_bool(i1 true)
  ; CHECK-NEXT: %z = icmp eq i1 %print, %1
  %z = icmp eq i1 %print, 1
  ; CHECK: %2 = call i1 @print_bool(i1 false)
  ; CHECK-NEXT: %y = icmp eq i1 %print, %2
  %y = icmp eq i1 %print, 0
  ret i1 %z
}

!revng.model = !{!1}
!0 = !{!"Isolated"}
!1 = !{!"---
...
"}
