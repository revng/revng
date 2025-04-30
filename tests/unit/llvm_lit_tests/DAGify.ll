;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -dagify -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; simple loop test

define void @f(i1 noundef %a) {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_d

block_d:
  br i1 %a, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: define void @f(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %a, label %goto_block_b, label %block_e
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b

; nested loops test

define void @g(i1 noundef %a, i1 noundef %b) {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_f

block_f:
  br label %block_g

block_g:
  br i1 %b, label %block_f, label %block_d

block_d:
  br i1 %a, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: define void @g(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   br i1 %b, label %goto_block_f, label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %a, label %goto_block_b, label %block_e
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_f:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_f
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b

; late entry loop test

define void @h(i1 noundef %a, i1 noundef %b) {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_c

block_c:
  br label %block_d

block_d:
  br i1 %b, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: define void @h(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %goto_block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %b, label %block_b, label %block_e
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_c:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_c

; nested loops same head test

define void @i(i1 noundef %a, i1 noundef %b) {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_f

block_f:
  br label %block_g

block_g:
  br i1 %b, label %block_b, label %block_d

block_d:
  br i1 %a, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: define void @i(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   br i1 %b, label %goto_block_b, label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %a, label %goto_block_b1, label %block_e
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b
; CHECK: goto_block_b1:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b

; nested loops same tail test

define void @l(i1 noundef %a) {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_f

block_f:
  br label %block_g

block_g:
  br label %block_d

block_d:
  %switch_val = zext i1 %a to i32
  switch i32 %switch_val, label %block_b [ i32 0, label %block_f
                                           i32 1, label %block_e ]

block_e:
  ret void
}

; CHECK-LABEL: define void @l(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   %switch_val = zext i1 %a to i32
; CHECK-NEXT:   switch i32 %switch_val, label %goto_block_b [
; CHECK-NEXT:     i32 0, label %goto_block_f
; CHECK-NEXT:     i32 1, label %block_e
; CHECK-NEXT:   ]
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_f:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_f
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b

; simple loop double retreating test

define void @m(i1 noundef %a) {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_d

block_d:
  %switch_val = zext i1 %a to i32
  switch i32 %switch_val, label %block_e [
    i32 0, label %block_b
    i32 1, label %block_b
  ]

block_e:
  ret void
}

; CHECK-LABEL: define void @m(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   %switch_val = zext i1 %a to i32
; CHECK-NEXT:   switch i32 %switch_val, label %block_e [
; CHECK-NEXT:     i32 0, label %goto_block_b
; CHECK-NEXT:     i32 1, label %goto_block_b
; CHECK-NEXT:   ]
; CHECK: block_e:
; CHECK-NEXT:   ret void
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b
