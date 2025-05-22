;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -inline-divergent-scopes -materialize-trivial-goto -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; trivial goto simplification enabled by correct IDS processing ordering

define void @f(i1 noundef %a, i1 noundef %b, i1 noundef %c) {

block_e:
  br i1 %a, label %block_c, label %block_a

block_a:
  br i1 %b, label %block_goto_y, label %block_goto_x

block_goto_y:
  call void @goto_block()
  br label %block_y

block_goto_x:
  call void @goto_block()
  br label %block_x

block_c:
  br label %block_x

block_x:
  br i1 %b, label %block_w, label %block_y

block_w:
  br label %block_p

block_y:
  br label %block_p

block_p:
  ret void
}

; CHECK-LABEL: define void @f(i1 noundef %a, i1 noundef %b, i1 noundef %c)
; CHECK: block_e:
; CHECK-NEXT:   br i1 %a, label %block_e_ids, label %block_a
; CHECK: block_a:
; CHECK-NEXT:   br i1 %b, label %block_goto_y, label %block_a_ids
; CHECK: block_goto_y:
; CHECK-NEXT:   call void @goto_block()
; CHECK-NEXT:   call void @scope_closer(ptr blockaddress(@f, %block_a_ids))
; CHECK-NEXT:   br label %block_y
; CHECK: block_c:
; CHECK-NEXT:   br label %block_x
; CHECK: block_x:
; CHECK-NEXT:   br i1 %b, label %block_w, label %block_y
; CHECK: block_w:
; CHECK-NEXT:   br label %block_p
; CHECK: block_y:
; CHECK-NEXT:   br label %block_p
; CHECK: block_p:
; CHECK-NEXT:   ret void
; CHECK: block_a_ids:
; CHECK-NEXT:   br label %block_x
; CHECK: block_e_ids:
; CHECK-NEXT:   br label %block_c
