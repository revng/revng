;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -materialize-trivial-goto -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; MTGoTo on if without scope closer

define void @f(i1 noundef %a) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
  br label %goto_block_d

goto_block_d:
  call void @goto_block()
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @f(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   ret void

; MTGoTo on if with a scope closer

define void @g(i1 noundef %a) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
br label %goto_block_d

goto_block_d:
  call void @scope_closer(ptr blockaddress(@g, %block_b))
  call void @goto_block()
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @g(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   ret void

; MTGoTo on autoloop (which should fail and rollback)

define void @h(i1 noundef %a) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
  call void @goto_block()
  br label %block_c

block_d:
  ret void
}

; CHECK-LABEL: define void @h(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_c:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_c
; CHECK: block_d:
; CHECK-NEXT:   ret void

; MTGoTo on autoloop with scope closer (which should fail and rollback)

define void @i(i1 noundef %a) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@i, %block_b))
  br label %block_c

block_d:
  ret void
}

; CHECK-LABEL: define void @i(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br label %block_d
; CHECK: block_c:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@i, %block_b))
; CHECK-NEXT:   br label %block_c
; CHECK: block_d:
; CHECK-NEXT:   ret void

; MTGoTo on scopegraph with two separately materializable gotos, but whose
; simplifications are mutually exclusive

define void @l(i1 noundef %a, i1 noundef %b) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_b:
  br i1 %b, label %block_d, label %goto_block_c

goto_block_c:
  call void @scope_closer(ptr blockaddress(@l, %block_d))
  call void @goto_block()
  br label %block_c

block_c:
  call void @scope_closer(ptr blockaddress(@l, %block_b))
  call void @goto_block()
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @l(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT:   br i1 %b, label %block_d, label %goto_block_c
; CHECK: goto_block_c:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@l, %block_d))
; CHECK-NEXT:   br label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   ret void
