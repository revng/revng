;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -remove-extractvalues | %revngopt -enable-new-pm=1 -O2 -S | FileCheck %s
; Test that all extractvalues are converted to an opaque function call that is
; not optimized away by the O2 pipeline

%0 = type { i64, i32, i16, i8}
; CHECK: %0 = type { i64, i32, i16, i8 }

; Function Attrs: noinline optnone
declare dso_local %0 @helper_returns_struct() local_unnamed_addr #0
; CHECK: declare dso_local %0 @helper_returns_struct() local_unnamed_addr #0

; Function Attrs: noinline optnone
declare dso_local %0 @save_i64() local_unnamed_addr #0

; Function Attrs: noinline optnone
declare dso_local %0 @save_i32() local_unnamed_addr #0

; Function Attrs: noinline optnone
declare dso_local %0 @save_i16() local_unnamed_addr #0

; Function Attrs: noinline optnone
declare dso_local %0 @save_i8() local_unnamed_addr #0

; Function Attrs: noreturn
define void @f() local_unnamed_addr #1 {
newFuncRoot:
  %0 = call %0 @helper_returns_struct()
  %1 = extractvalue %0 %0, 0     ; CHECK: tail call i64 @{{OpaqueExtractvalue.[0-9]+|OpaqueExtractvalue}}(%0 %0, i64 0)
  %2 = extractvalue %0 %0, 1     ; CHECK-NEXT: tail call i32 @{{OpaqueExtractvalue.[0-9]+|OpaqueExtractvalue}}(%0 %0, i64 1)
  %3 = extractvalue %0 %0, 2     ; CHECK-NEXT: tail call i16 @{{OpaqueExtractvalue.[0-9]+|OpaqueExtractvalue}}(%0 %0, i64 2)
  %4 = extractvalue %0 %0, 3     ; CHECK-NEXT: tail call i8 @{{OpaqueExtractvalue.[0-9]+|OpaqueExtractvalue}}(%0 %0, i64 3)
  call void @save_i64(i64 %1)
  call void @save_i32(i32 %2)
  call void @save_i16(i16 %3)
  call void @save_i8(i8 %4)
  unreachable
}

attributes #0 = { noinline optnone }
attributes #1 = { noreturn }
