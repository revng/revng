;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -inline-divergent-scopes -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope-closer(ptr)
declare !revng.tags !1 void @goto-block()
!0 = !{!"scope-closer"}
!1 = !{!"goto-block"}

; IDS on if test

define void @f(i1 noundef %a, i1 noundef %b) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_c:
  br i1 %b, label %block_b, label %block_e

block_b:
  ret void

block_e:
  ret void
}

; CHECK-LABEL: define void @f(i1 noundef %a, i1 noundef %b)
; CHECK: block_a:
; CHECK-NEXT:   br i1 %a, label %block_b, label %block_c
; CHECK: block_c:
; CHECK-NEXT:   br i1 %b, label %block_c_ids, label %block_e
; CHECK: block_b:
; CHECK-NEXT:   ret void
; CHECK: block_e:
; CHECK-NEXT:   call void @scope-closer(ptr blockaddress(@f, %block_c_ids))
; CHECK-NEXT:   ret void
; CHECK: block_c_ids:
; CHECK-NEXT:   br label %block_b

define void @g(i1 noundef %a, i32 noundef %b) {

block_a:
  br i1 %a, label %block_b, label %block_c

block_c:
  switch i32 %b, label %block_b [
    i32 0, label %block_e
    i32 1, label %block_f
  ]

block_b:
  ret void

block_e:
  ret void

block_f:
  ret void
}

; CHECK: define void @g(i1 noundef %a, i32 noundef %b)
; CHECK: block_a:
; CHECK:   br i1 %a, label %block_b, label %block_c
; CHECK: block_c:
; CHECK:   switch i32 %b, label %block_c_ids [
; CHECK:     i32 0, label %block_e
; CHECK:     i32 1, label %block_c_ids
; CHECK:   ]
; CHECK: block_b:
; CHECK:   ret void
; CHECK: block_e:
; CHECK:   call void @scope-closer(ptr blockaddress(@g, %block_c_ids))
; CHECK:   ret void
; CHECK: block_f:
; CHECK:   call void @scope-closer(ptr blockaddress(@g, %block_c_ids_ids))
; CHECK:   ret void
; CHECK: block_c_ids:
; CHECK:   switch i32 %b, label %block_c_ids_ids [
; CHECK:     i32 1, label %block_f
; CHECK:   ]
; CHECK: block_c_ids_ids:
; CHECK:   switch i32 %b, label %block_b [
; CHECK:   ]
