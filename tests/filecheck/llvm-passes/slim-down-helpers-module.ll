;
; This file is distributed under the MIT License. See LICENSE.md for details.
;
; Check that `slim-down-helpers-module` drops the initializers of QEMU's
; globals, but keeps the strings emitted by rev.ng, such as the messages of
; the aborts emitted by `fix-helpers`.
;
; RUN: %root/bin/revng opt -slim-down-helpers-module %s -S -o - | FileCheck %s

@qemu_table = internal constant [4 x i8] c"\00\01\02\03"
; CHECK-DAG: @qemu_table = external {{.*}}constant [4 x i8]{{,|$}}

@revng.const.Abort = linkonce_odr constant [6 x i8] c"Abort\00"
; CHECK-DAG: @revng.const.Abort = linkonce_odr constant [6 x i8] c"Abort\00"

declare void @revng_abort(ptr)

define i8 @helper_test(i64 %index) section "revng_inline" {
entry:
  %in_range = icmp ult i64 %index, 4
  br i1 %in_range, label %lookup, label %abort
lookup:
  %p = getelementptr [4 x i8], ptr @qemu_table, i64 0, i64 %index
  %v = load i8, ptr %p
  ret i8 %v
abort:
  call void @revng_abort(ptr @revng.const.Abort)
  unreachable
}
