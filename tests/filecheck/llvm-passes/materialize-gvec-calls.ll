;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; Use the installed QEMU helper body. The pass must expose register effects
; without leaving temporary storage or helpers for function outlining.
;
; RUN: %root/bin/revng opt -materialize-gvec-calls -verify %s -S -o - | FileCheck %s --implicit-check-not=alloca --implicit-check-not='{{call[[:space:]]}}'
; RUN: %root/bin/revng opt -materialize-gvec-calls -materialize-gvec-calls -verify %s -S -o - | FileCheck %s --implicit-check-not=alloca --implicit-check-not='{{call[[:space:]]}}'

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

@r0 = global i64 0, !revng.csv.offset !0
@r1 = global i64 0, !revng.csv.offset !1
@r2 = global i64 0, !revng.csv.offset !2
@r3 = global i64 0, !revng.csv.offset !3
@r4 = global i64 0, !revng.csv.offset !4
@r5 = global i64 0, !revng.csv.offset !5
@r6 = global i64 0, !revng.csv.offset !6
@r7 = global i64 0, !revng.csv.offset !7
@z0 = global i64 0, !revng.csv.offset !9
@z1 = global i64 0, !revng.csv.offset !10

declare void @helper_gvec_eq8(ptr writeonly, ptr readonly, ptr readonly, i32)
declare void @helper_gvec_add8(ptr writeonly, ptr readonly, ptr readonly, i32)
declare void @helper_gvec_dup64(ptr writeonly, i32, i64)
declare void @helper_gvec_adds64(ptr writeonly, ptr readonly, i64, i32)

; CHECK-LABEL: define void @equal()
; CHECK-NOT: alloca
; CHECK-NOT: call
; CHECK: load i64, ptr @r2
; CHECK: icmp eq
; CHECK: store i64 {{.*}}, ptr @r0
; CHECK: ret void
define void @equal() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), ptr inttoptr (i64 544 to ptr), i32 513)
  ret void
}

; CHECK-LABEL: define void @alias()
; CHECK-NOT: alloca
; CHECK-NOT: call
; CHECK: icmp eq
; CHECK: ret void
define void @alias() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), i32 513)
  ret void
}

; Partial overlap checks byte ownership across CPU-state fields.
; CHECK-LABEL: define void @overlap()
; CHECK-NOT: alloca
; CHECK-NOT: call
; CHECK: icmp eq
; CHECK: ret void
define void @overlap() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 516 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), i32 513)
  ret void
}

; The descriptor requests 16 active bytes and a 32-byte destination.
; CHECK-LABEL: define void @tail()
; CHECK-NOT: alloca
; CHECK-NOT: call
; CHECK: store i64 0, ptr @r2
; CHECK: store i64 0, ptr @r3
; CHECK: ret void
define void @tail() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 544 to ptr), ptr inttoptr (i64 560 to ptr), i32 259)
  ret void
}

; A distinct operation uses the same specialization, with no opcode rewrite.
; CHECK-LABEL: define void @add()
; CHECK-NOT: alloca
; CHECK-NOT: call
; CHECK: add
; CHECK: ret void
define void @add() {
  call void @helper_gvec_add8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), ptr inttoptr (i64 544 to ptr), i32 513)
  ret void
}

; This helper places its descriptor before the scalar argument. Offset
; zero is a CPU-state address too, and a full write must not read the previous
; destination value.
; CHECK-LABEL: define void @duplicate()
; CHECK-NOT: load
; CHECK: store i64 123, ptr @z0
; CHECK-NOT: load
; CHECK: store i64 123, ptr @z1
; CHECK-NEXT: ret void
define void @duplicate() {
  call void @helper_gvec_dup64(ptr null, i32 513, i64 123)
  ret void
}

; CHECK-LABEL: define void @add_scalar(i64 %value)
; CHECK: add i64 {{.*}}%value
; CHECK: store i64 {{.*}}, ptr @z0
; CHECK: add i64 {{.*}}%value
; CHECK: store i64 {{.*}}, ptr @z1
; CHECK-NEXT: ret void
define void @add_scalar(i64 %value) {
  call void @helper_gvec_adds64(ptr null, ptr inttoptr (i64 512 to ptr), i64 %value, i32 513)
  ret void
}

!revng.qemu_architecture = !{!8}
!0 = !{i64 512}
!1 = !{i64 520}
!2 = !{i64 528}
!3 = !{i64 536}
!4 = !{i64 544}
!5 = !{i64 552}
!6 = !{i64 560}
!7 = !{i64 568}
!8 = !{!"x86_64"}
!9 = !{i64 0}
!10 = !{i64 8}
