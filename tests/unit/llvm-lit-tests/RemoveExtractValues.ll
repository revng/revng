;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -remove-extractvalues | %revngopt -enable-new-pm=1 -O2 -S | FileCheck %s
; Test that all extractvalues are converted to an opaque function call that is
; not optimized away by the O2 pipeline

%0 = type { i64, i64, i64, i32, i64, i64, i32, i64, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, i32, i64, i32, i64, i32, i64 }
%struct.CPUX86State = type { i64, i32 }

; Function Attrs: noinline optnone
declare dso_local %0 @helper_syscall_wrapper(%struct.CPUX86State*, i32, i64, i64, i32, i64, i64, i64, i64, i64, i32, i64, i32, i32, i64, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) local_unnamed_addr #0

; Function Attrs: noinline nounwind optnone readonly willreturn
declare i32 @init_i32() local_unnamed_addr #1

; Function Attrs: noinline nounwind optnone readonly willreturn
declare i64 @init_i64() local_unnamed_addr #1

; Function Attrs: noreturn
define void @local__Exit(i64 %rdi) local_unnamed_addr #2 {
newFuncRoot:
  %0 = tail call i64 @init_i64()
  %1 = tail call i32 @init_i32()
  %sext88103233 = shl i64 %rdi, 32
  %2 = ashr exact i64 %sext88103233, 32
  %3 = tail call %0 @helper_syscall_wrapper(%struct.CPUX86State* null, i32 2, i64 4206331, i64 %0, i32 %1, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0, i32 %1, i64 %0, i32 %1, i32 %1, i64 %0, i32 %1, i32 %1, i32 %1, i32 %1, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0, i64 231, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0, i64 %2, i64 %0)
  %4 = extractvalue %0 %3, 2     ; CHECK: tail call i64 @OpaqueExtractvalue(%0 %3, i64 2)
  %5 = extractvalue %0 %3, 3     ; CHECK-NEXT: tail call i32 @OpaqueExtractvalue.1(%0 %3, i64 3)
  %6 = extractvalue %0 %3, 4     ; CHECK-NEXT: tail call i64 @OpaqueExtractvalue(%0 %3, i64 4)
  %7 = extractvalue %0 %3, 5     ; CHECK-NEXT: tail call i64 @OpaqueExtractvalue(%0 %3, i64 5)
  %8 = extractvalue %0 %3, 6     ; CHECK-NEXT: tail call i32 @OpaqueExtractvalue.1(%0 %3, i64 6)
  %9 = extractvalue %0 %3, 7     ; CHECK-NEXT: tail call i64 @OpaqueExtractvalue(%0 %3, i64 7)
  %10 = extractvalue %0 %3, 8    ; CHECK-NEXT: tail call i32 @OpaqueExtractvalue.1(%0 %3, i64 8)
  %11 = extractvalue %0 %3, 9    ; CHECK-NEXT: tail call i32 @OpaqueExtractvalue.1(%0 %3, i64 9)
  %12 = extractvalue %0 %3, 10   ; CHECK-NEXT: tail call i32 @OpaqueExtractvalue.1(%0 %3, i64 10)
  %13 = extractvalue %0 %3, 23   ; CHECK-NEXT: tail call i64 @OpaqueExtractvalue(%0 %3, i64 23)
  br label %bb._Exit.0xf3234

bb._Exit.0xf3234:                                 ; preds = %bb._Exit.0xf3234, %newFuncRoot
  %state_0x8300.0 = phi i32 [ %12, %newFuncRoot ], [ %23, %bb._Exit.0xf3234 ]
  %state_0x9080.0 = phi i32 [ %11, %newFuncRoot ], [ %22, %bb._Exit.0xf3234 ]
  %state_0x83a8.0 = phi i32 [ %10, %newFuncRoot ], [ %21, %bb._Exit.0xf3234 ]
  %state_0x83a0.0 = phi i64 [ %9, %newFuncRoot ], [ %20, %bb._Exit.0xf3234 ]
  %state_0x8248.0 = phi i32 [ %8, %newFuncRoot ], [ %19, %bb._Exit.0xf3234 ]
  %state_0x82d8.0 = phi i64 [ %7, %newFuncRoot ], [ %18, %bb._Exit.0xf3234 ]
  %state_0x8388.0 = phi i64 [ %6, %newFuncRoot ], [ %17, %bb._Exit.0xf3234 ]
  %state_0x9010.0 = phi i32 [ %5, %newFuncRoot ], [ %16, %bb._Exit.0xf3234 ]
  %state_0x9018.0 = phi i64 [ %4, %newFuncRoot ], [ %15, %bb._Exit.0xf3234 ]
  %state_0x8370.0 = phi i64 [ %13, %newFuncRoot ], [ %24, %bb._Exit.0xf3234 ]
  %14 = tail call %0 @helper_syscall_wrapper(%struct.CPUX86State* null, i32 2, i64 4206341, i64 %state_0x9018.0, i32 %state_0x9010.0, i64 %0, i64 %0, i64 %state_0x8388.0, i64 %0, i64 %state_0x82d8.0, i32 %state_0x8248.0, i64 %state_0x83a0.0, i32 %state_0x83a8.0, i32 %1, i64 %0, i32 %state_0x9080.0, i32 %state_0x8300.0, i32 %1, i32 %1, i64 %0, i64 %0, i64 %0, i64 %0, i64 %0, i64 60, i64 %0, i64 %0, i64 60, i64 %0, i64 %0, i64 %0, i64 %2, i64 %state_0x8370.0)
  %15 = extractvalue %0 %14, 2   ; CHECK:        tail call i64 @OpaqueExtractvalue(%0 %14, i64 2)
  %16 = extractvalue %0 %14, 3   ; CHECK-NEXT:   tail call i32 @OpaqueExtractvalue.1(%0 %14, i64 3)
  %17 = extractvalue %0 %14, 4   ; CHECK-NEXT:   tail call i64 @OpaqueExtractvalue(%0 %14, i64 4)
  %18 = extractvalue %0 %14, 5   ; CHECK-NEXT:   tail call i64 @OpaqueExtractvalue(%0 %14, i64 5)
  %19 = extractvalue %0 %14, 6   ; CHECK-NEXT:   tail call i32 @OpaqueExtractvalue.1(%0 %14, i64 6)
  %20 = extractvalue %0 %14, 7   ; CHECK-NEXT:   tail call i64 @OpaqueExtractvalue(%0 %14, i64 7)
  %21 = extractvalue %0 %14, 8   ; CHECK-NEXT:   tail call i32 @OpaqueExtractvalue.1(%0 %14, i64 8)
  %22 = extractvalue %0 %14, 9   ; CHECK-NEXT:   tail call i32 @OpaqueExtractvalue.1(%0 %14, i64 9)
  %23 = extractvalue %0 %14, 10   ; CHECK-NEXT:  tail call i32 @OpaqueExtractvalue.1(%0 %14, i64 10)
  %24 = extractvalue %0 %14, 23   ; CHECK-NEXT:  tail call i64 @OpaqueExtractvalue(%0 %14, i64 23)
  br label %bb._Exit.0xf3234
}

attributes #0 = { noinline optnone }
attributes #1 = { noinline nounwind optnone readonly willreturn }
attributes #2 = { noreturn }
