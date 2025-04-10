;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -select-scope -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope-closer(ptr)
declare !revng.tags !1 void @goto-block()
!0 = !{!"scope-closer"}
!1 = !{!"goto-block"}

; decided diamond test

define void @f(i1 noundef %a) {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_d:
  br label %block_f

block_c:
  br label %block_e

block_e:
  br label %block_f

block_f:
  ret void
}

; CHECK-LABEL: define void @f(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br label %block_f
; CHECK: block_c:
; CHECK-NEXT:   br label %block_e
; CHECK: block_e:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   ret void

; crossed diamonds test

define void @g(i1 noundef %a, i1 noundef %b, i1 noundef %c) {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br i1 %b, label %block_d, label %block_e

block_d:
  br label %block_f

block_c:
  br i1 %c, label %block_e, label %block_d

block_e:
  br label %block_f

block_f:
  ret void
}

; CHECK-LABEL: define void @g(i1 noundef %a, i1 noundef %b, i1 noundef %c)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br i1 %b, label %block_d, label %block_e
; CHECK: block_d:
; CHECK-NEXT:   br label %block_f
; CHECK: block_c:
; CHECK-NEXT:   br i1 %c, label %goto_block_e, label %goto_block_d
; CHECK: block_e:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   ret void
; CHECK: goto_block_e:
; CHECK-NEXT:   call void @goto-block()
; CHECK-NEXT:   br label %block_e
; CHECK: goto_block_d:
; CHECK-NEXT:   call void @goto-block()
; CHECK-NEXT:   br label %block_d

; double edge if test

define void @h(i1 noundef %a) {
block_a:
  br i1 %a, label %block_b, label %block_b

block_b:
  br label %block_c

block_c:
  ret void
}

; CHECK-LABEL: define void @h(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_b
; CHECK: block_b:
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   ret void

; double edge switch test

define void @i(i1 noundef %a) {
block_a:
  switch i1 %a, label %block_b [
    i1 0, label %block_c
    i1 1, label %block_c
  ]

block_b:
  br label %block_d

block_c:
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @i(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   switch i1 %a, label %block_b [
; CHECK-NEXT:     i1 false, label %block_c
; CHECK-NEXT:     i1 true, label %block_c
; CHECK-NEXT:   ]
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   ret void

; short-circuit test

define void @l(i1 noundef %a, i1 noundef %b) {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br i1 %b, label %block_d, label %block_e

block_d:
  br label %block_f

block_c:
  br label %block_e

block_e:
  br label %block_f

block_f:
  ret void
}

; CHECK-LABEL: define void @l(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br i1 %b, label %block_d, label %block_e
; CHECK: block_d:
; CHECK-NEXT:   br label %block_f
; CHECK: block_c:
; CHECK-NEXT:   br label %goto_block_e
; CHECK: block_e:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   ret void
; CHECK: goto_block_e:
; CHECK-NEXT:   call void @goto-block()
; CHECK-NEXT:   br label %block_e

; short-circuit with decided tail

define void @m(i1 noundef %a, i1 noundef %b, i1 noundef %c) {
block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br i1 %b, label %block_d, label %block_e

block_d:
  br label %block_f

block_c:
  br label %block_e

block_e:
  br label %block_f

block_f:
  br label %block_g

block_g:
  br i1 %c, label %block_h, label %block_i

block_h:
  br label %block_l

block_i:
  br label %block_l

block_l:
  ret void
}

; CHECK: define void @m(i1 noundef %a, i1 noundef %b, i1 noundef %c)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br i1 %b, label %block_d, label %block_e
; CHECK: block_d:
; CHECK-NEXT:   br label %block_f
; CHECK: block_c:
; CHECK-NEXT:   br label %goto_block_e
; CHECK: block_e:
; CHECK-NEXT:   br label %block_f
; CHECK: block_f:
; CHECK-NEXT:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   br i1 %c, label %block_h, label %block_i
; CHECK: block_h:
; CHECK-NEXT:   br label %block_l
; CHECK: block_i:
; CHECK-NEXT:   br label %block_l
; CHECK: block_l:
; CHECK-NEXT:   ret void
; CHECK: goto_block_e:
; CHECK-NEXT:   call void @goto-block()
; CHECK-NEXT:   br label %block_e
