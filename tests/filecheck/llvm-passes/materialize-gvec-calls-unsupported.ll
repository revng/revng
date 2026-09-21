;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; Unsupported calls retain their original operands and produce a diagnostic.
; A failure must not prevent the same module's supported calls from lowering.
;
; RUN: %root/bin/revng opt -materialize-gvec-calls -verify %s -S -o %t 2>%t.err
; RUN: FileCheck %s --input-file=%t --implicit-check-not=alloca --implicit-check-not=scalarized_gvec
; RUN: FileCheck %s --check-prefix=DIAG --input-file=%t.err

target datalayout = "e-m:e-p:64:64-i64:64-n8:16:32:64"

@r0 = global i64 0, !revng.csv.offset !0
@r1 = global i64 0, !revng.csv.offset !1
@r2 = global i64 0, !revng.csv.offset !2
@r3 = global i64 0, !revng.csv.offset !3

declare void @helper_gvec_eq8(ptr writeonly, ptr readonly, ptr readonly, i32)
declare void @helper_gvec_dup64(ptr writeonly, i32, i64)

; DIAG: retaining helper_gvec_eq8: pointer is not a constant CPU-state offset
; CHECK-LABEL: define void @dynamic_pointer(ptr %source)
; CHECK: call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr %source, ptr inttoptr (i64 528 to ptr), i32 513)
define void @dynamic_pointer(ptr %source) {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr %source, ptr inttoptr (i64 528 to ptr), i32 513)
  ret void
}

; DIAG: retaining helper_gvec_eq8: helper access is not a constant integer CPU-state access
; CHECK-LABEL: define void @dynamic_descriptor(i32 %descriptor)
; CHECK: call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), i32 %descriptor)
define void @dynamic_descriptor(i32 %descriptor) {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), i32 %descriptor)
  ret void
}

; The pointer's first byte is mapped, but the access extends past the last CSV.
; The temporary CPU-state base must not impose a size that turns this access
; into undefined behavior before the pass can reject it.
; DIAG: retaining helper_gvec_eq8: helper access crosses unmapped CPU-state bytes
; CHECK-LABEL: define void @unmapped_tail()
; CHECK: call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 532 to ptr), i32 513)
define void @unmapped_tail() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 532 to ptr), i32 513)
  ret void
}

; A constant descriptor does not guarantee constant memory accesses: dup64's
; zero-value shortcut changes where its clear-high loop starts.
; DIAG: retaining helper_gvec_dup64: helper access is not a constant integer CPU-state access
; CHECK-LABEL: define void @dynamic_duplicate(i64 %value)
; CHECK: call void @helper_gvec_dup64(ptr inttoptr (i64 512 to ptr), i32 513, i64 %value)
define void @dynamic_duplicate(i64 %value) {
  call void @helper_gvec_dup64(ptr inttoptr (i64 512 to ptr), i32 513, i64 %value)
  ret void
}

; CHECK-LABEL: define void @supported()
; CHECK-NOT: call
; CHECK: icmp eq
; CHECK: store i64 {{.*}}, ptr @r0
; CHECK-NOT: call
; CHECK: ret void
define void @supported() {
  call void @helper_gvec_eq8(ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 512 to ptr), ptr inttoptr (i64 528 to ptr), i32 513)
  ret void
}

!revng.qemu_architecture = !{!4}
!0 = !{i64 512}
!1 = !{i64 520}
!2 = !{i64 528}
!3 = !{i64 536}
!4 = !{!"x86_64"}
