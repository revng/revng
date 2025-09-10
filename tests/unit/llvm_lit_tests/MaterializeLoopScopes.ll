;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -materialize-loop-scopes -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}
!2 = !{}

; MLoopScopes on not dagified loop with single successor

define void @f(i1 noundef %a) {

block_a:
  br label %block_b

block_b:
  br label %block_c, !genericregion-head !2

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
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@f, %block_e))
; CHECK-NEXT:   br label %loop_start
; CHECK: loop_start:
; CHECK-NEXT:   br label %block_c, !genericregion-head !2
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_e
; CHECK: block_e:
; CHECK-NEXT:   ret void

; MLoopScopes on dagified loop with single successor

define void @g(i1 noundef %a) {

block_a:
  br label %block_b

block_b:
  br label %block_c, !genericregion-head !2

block_c:
  br label %block_d

block_d:
  br i1 %a, label %goto_block_b, label %block_e

goto_block_b:
  call void @scope_closer(ptr blockaddress(@g, %block_e))
  call void @goto_block()
  br label %block_b

block_e:
  ret void
}

; CHECK-LABEL: define void @g(i1 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@g, %block_e))
; CHECK-NEXT:   br label %loop_start
; CHECK: loop_start:
; CHECK-NEXT:   br label %block_c, !genericregion-head !2
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br i1 %a, label %goto_block_b, label %block_e
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@g, %block_e))
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   br label %block_b
; CHECK: block_e:
; CHECK-NEXT:   ret void

; MLoopScopes on not dagified loop with multiple successors, footer insertion

define void @h(i32 noundef %a) {

block_a:
  br label %block_b

block_b:
  switch i32 %a, label %block_e [
    i32 0, label %block_f
    i32 1, label %block_c
  ], !genericregion-head !2

block_c:
  br label %block_d

block_d:
  br label %block_b

block_e:
  br label %block_g

block_f:
  br label %block_g

block_g:
  ret void
}

; CHECK-LABEL: define void @h(i32 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@h, %footer))
; CHECK-NEXT:   br label %loop_start
; CHECK: loop_start:
; CHECK-NEXT:   switch i32 %a, label %footer [
; CHECK-NEXT:     i32 0, label %footer
; CHECK-NEXT:     i32 1, label %block_c
; CHECK-NEXT:   ], !genericregion-head !2
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br label %block_b
; CHECK: block_e:
; CHECK-NEXT:   br label %block_g
; CHECK: block_f:
; CHECK-NEXT:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   ret void
; CHECK: footer:
; CHECK-NEXT:   switch i32 %a, label %block_e [
; CHECK-NEXT:     i32 0, label %block_f
; CHECK-NEXT:   ], !genericregion-head !2

; MLoopScopes on dagified loop with multiple successors, no footer is inserted

define void @i(i32 noundef %a) {
block_a:
  br label %block_b

block_b:
  switch i32 %a, label %block_b_ids [
    i32 0, label %block_b_ids
    i32 1, label %block_c
  ], !genericregion-head !2

block_c:
  br label %block_d

block_d:
  br label %goto_block_b

block_e:
  br label %block_g

block_f:
  br label %block_g

block_g:
  ret void

goto_block_b:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@i, %block_b_ids))
  br label %block_b

block_b_ids:
  switch i32 %a, label %block_e [
    i32 0, label %block_f
  ]
}

; CHECK-LABEL: define void @i(i32 noundef %a)
; CHECK: block_a:
; CHECK-NEXT:   br label %block_b
; CHECK: block_b:
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@i, %block_b_ids))
; CHECK-NEXT:   br label %loop_start
; CHECK: loop_start:
; CHECK-NEXT:   switch i32 %a, label %block_b_ids [
; CHECK-NEXT:     i32 0, label %block_b_ids
; CHECK-NEXT:     i32 1, label %block_c
; CHECK-NEXT:   ], !genericregion-head !2
; CHECK: block_c:
; CHECK-NEXT:   br label %block_d
; CHECK: block_d:
; CHECK-NEXT:   br label %goto_block_b
; CHECK: block_e:
; CHECK-NEXT:   br label %block_g
; CHECK: block_f:
; CHECK-NEXT:   br label %block_g
; CHECK: block_g:
; CHECK-NEXT:   ret void
; CHECK: goto_block_b:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@i, %block_b_ids))
; CHECK-NEXT:   br label %block_b
; CHECK: block_b_ids:
; CHECK-NEXT:   switch i32 %a, label %block_e [
; CHECK-NEXT:     i32 0, label %block_f
; CHECK-NEXT:   ]
