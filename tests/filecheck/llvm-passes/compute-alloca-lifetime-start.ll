;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -compute-alloca-lifetime-start -S -o - | FileCheck %s

; The pass marks, with a `llvm.lifetime.start`, the point dominating all the
; uses of each alloca, which is where the clifter declares the corresponding
; local variable. The alloca itself is not moved.
;
; The pass expects no PHINodes, which `exit-ssa` guarantees where it runs, and
; asserts it. All the functions here are therefore free of them.

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @opaque(i64)

; With every use confined to one block, the marker goes there, before the first
; of them, while the alloca stays where it was.

define void @inner_uses(i1 %c) {
entry:
  %p = alloca i64
  br i1 %c, label %inner, label %other

inner:
  store i64 1, ptr %p
  %v = load i64, ptr %p
  call void @opaque(i64 %v)
  br label %exit

other:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @inner_uses(i1 %c)
; CHECK: entry:
; CHECK-NEXT: %p = alloca i64
; CHECK-NEXT: br i1 %c, label %inner, label %other
; CHECK: inner:
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %p)
; CHECK-NEXT: store i64 1, ptr %p

; With uses in two arms, the marker goes to the block dominating both. Nothing
; uses the variable there, so it goes at the end of it.

define void @two_siblings(i1 %c) {
entry:
  %p = alloca i64
  br i1 %c, label %left, label %right

left:
  store i64 1, ptr %p
  br label %exit

right:
  store i64 2, ptr %p
  br label %exit

exit:
  %v = load i64, ptr %p
  call void @opaque(i64 %v)
  ret void
}

; CHECK-LABEL: define void @two_siblings(i1 %c)
; CHECK: entry:
; CHECK-NEXT: %p = alloca i64
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %p)
; CHECK-NEXT: br i1 %c, label %left, label %right

; A variable `exit-ssa` already marked is left alone. Its marker is placed from
; the class of PHIs the variable stood for, which puts it outside any loop the
; variable is carried across, and that is better than anything derivable from
; the accesses alone.

define void @already_marked() {
entry:
  %p = alloca i64
  call void @llvm.lifetime.start.p0(i64 -1, ptr %p)
  br label %body

body:
  store i64 1, ptr %p
  ret void
}

; CHECK-LABEL: define void @already_marked()
; CHECK: entry:
; CHECK-NEXT: %p = alloca i64
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %p)
; CHECK-NEXT: br label %body
; CHECK: body:
; CHECK-NOT: lifetime.start
; CHECK: store i64 1, ptr %p

; The stack frame is live for the whole function and the clifter gives it a
; declaration of its own, so it gets no marker.

define void @stack_frame() {
entry:
  %p = alloca [16 x i8], !revng.stack_frame !0

  br label %body

body:
  store i64 1, ptr %p
  ret void
}

; CHECK-LABEL: define void @stack_frame()
; CHECK-NOT: lifetime.start
; CHECK: store i64 1, ptr %p

; An alloca nothing uses has no point at which it becomes live, and is left to
; the clifter, which declares it where the alloca sits.

define void @unused() {
entry:
  %p = alloca i64
  ret void
}

; CHECK-LABEL: define void @unused()
; CHECK-NOT: lifetime.start
; CHECK: ret void

!0 = !{}
