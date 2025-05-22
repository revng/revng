;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -enforce-single-exit -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; two trivial exits test

define void @f() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  br i1 undef, label %block_b, label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @f
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT: ret void
; CHECK: block_c:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_d
; CHECK: block_d:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@f, %block_b))
; CHECK-NEXT: ret void

; one trivial, one non-trivial exit test

define void @g() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  br label %block_d

block_d:
  br label %block_c
}

; CHECK-LABEL: define void @g
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT: ret void
; CHECK: block_c:
; CHECK-NEXT: br label %block_d
; CHECK: block_d:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@g, %block_b))
; CHECK-NEXT: br label %block_c

; one trivial, one non-trivial exit (self loop) test

define void @h() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  br label %block_c
}

; CHECK-LABEL: define void @h
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT: ret void
; CHECK: block_c:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@h, %block_b))
; CHECK-NEXT: br label %block_c

; two trivial exit, with one scope_closer edge already present test

define void @i() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  call void @scope_closer(ptr blockaddress(@i, %block_b))
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @i
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT: ret void
; CHECK: block_c:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@i, %block_b))
; CHECK-NEXT: br label %block_d
; CHECK: block_d:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@i, %block_b))
; CHECK-NEXT: ret void

; SCC composed by a `scope_closer` edge test

define void @l() {
block_a:
  br i1 undef, label %block_c, label %block_b

block_b:
  br i1 undef, label %block_e, label %block_d

block_c:
  call void @scope_closer(ptr blockaddress(@l, %block_d))
  unreachable

block_d:
  br label %block_c

block_e:
  ret void
}

; CHECK-LABEL: define void @l
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_c, label %block_b
; CHECK: block_b:
; CHECK-NEXT: br i1 undef, label %block_e, label %block_d
; CHECK: block_c:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@l, %block_d))
; CHECK-NEXT: unreachable
; CHECK: block_d:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@l, %block_e))
; CHECK-NEXT: br label %block_c
; CHECK: block_e:
; CHECK-NEXT: ret void

; no trivial exit, one non-trivial exit

define void @m() {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_b
}

; CHECK-LABEL: define void @m
; CHECK: new_entry_block:
; CHECK-NEXT: br i1 true, label %block_a, label %sink_block
; CHECK: block_a:
; CHECK-NEXT: br label %block_b
; CHECK: block_b:
; CHECK-NEXT: br label %block_c
; CHECK: block_c:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@m, %sink_block))
; CHECK-NEXT: br label %block_b
; CHECK: sink_block:
; CHECK-NEXT: unreachable

; two trivial exits

define void @n() {
block_a:
  br i1 true, label %block_b, label %block_d

block_b:
  br label %block_c

block_c:
  br label %block_b

block_d:
  br label %block_e

block_e:
  br label %block_d
}

; CHECK-LABEL: define void @n
; CHECK: new_entry_block:
; CHECK-NEXT: br i1 true, label %block_a, label %sink_block
; CHECK: block_a:
; CHECK-NEXT: br i1 true, label %block_b, label %block_d
; CHECK: block_b:
; CHECK-NEXT: br label %block_c
; CHECK: block_c:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@n, %sink_block))
; CHECK-NEXT: br label %block_b
; CHECK: block_d:
; CHECK-NEXT: br label %block_e
; CHECK: block_e:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@n, %sink_block))
; CHECK-NEXT: br label %block_d
; CHECK: sink_block:
; CHECK-NEXT: unreachable

; one trivial exit, no modification test

define void @o() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  br label %block_d

block_c:
  br label %block_d

block_d:
  ret void
}

; CHECK-LABEL: define void @o
; CHECK: block_a:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_c
; CHECK: block_b:
; CHECK-NEXT: br label %block_d
; CHECK: block_c:
; CHECK-NEXT: br label %block_d
; CHECK: block_d:
; CHECK-NEXT: ret void

; two nested sccs

define void @p() {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br i1 undef, label %block_b, label %block_d

block_d:
  br label %block_b
}

; CHECK-LABEL: define void @p
; CHECK: new_entry_block:
; CHECK-NEXT: br i1 true, label %block_a, label %sink_block
; CHECK: block_a:
; CHECK-NEXT: br label %block_b
; CHECK: block_b:
; CHECK-NEXT: br label %block_c
; CHECK: block_c:
; CHECK-NEXT: br i1 undef, label %block_b, label %block_d
; CHECK: block_d:
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@p, %sink_block))
; CHECK-NEXT: br label %block_b
; CHECK: sink_block:
; CHECK-NEXT: unreachable
