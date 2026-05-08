;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt %s -arithmetic-to-gep -S -o - | FileCheck %s

; =============================================================================
; =============================================================================
; Simple tests to check that various kinds of integer arithmetic are converted
; to GEPs, without the !revng.pointers metadata.
; =============================================================================
; =============================================================================

; ----------------------------------------------------
; Functions with pointer argument, integer return type
; ----------------------------------------------------

; Pointer argument, cast to integer, then incremented and returned as integer
;
; CHECK-LABEL: define i64 @a
; CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %arg, i64 1
; CHECK-NEXT: [[CAST:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[CAST]]
define i64 @a (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, 1
  ret i64 %with_offset
}

; Same as above, but with more casts back and forth between integer and pointer
;
; CHECK-LABEL: define i64 @b
; CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %arg, i64 2
; CHECK-NEXT: [[CAST:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[CAST]]
define i64 @b (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %a = inttoptr i64 %intptr to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  %c = inttoptr i64 %with_offset to ptr
  %d = ptrtoint ptr %c to i64
  ret i64 %d
}

; Same as @a, but check that the transformation works even with unknown offset
;
; CHECK-LABEL: define i64 @unknown_offset
; CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %arg, i64 %offset
; CHECK-NEXT: [[CAST:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[CAST]]
define i64 @unknown_offset (ptr %arg, i64 %offset) {
  %intptr = ptrtoint ptr %arg to i64
  %with_offset = add i64 %intptr, %offset
  ret i64 %with_offset
}

; ----------------------------------------------------
; Functions with integer argument, pointer return type
; ----------------------------------------------------

; Pointer argument, cast to integer, then incremented and returned as integer
;
; CHECK-LABEL: define ptr @aa
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 1
; CHECK-NEXT: ret ptr [[GEP]]
define ptr @aa (i64 %arg) {
  %with_offset = add i64 %arg, 1
  %ptr = inttoptr i64 %with_offset to ptr
  ret ptr %ptr
}

; Same as above, but with more casts back and forth between integer and pointer
;
; CHECK-LABEL: define ptr @bb
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 2
; CHECK-NEXT: ret ptr [[GEP]]
define ptr @bb (i64 %arg) {
  %a = inttoptr i64 %arg to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  %c = inttoptr i64 %with_offset to ptr
  %d = ptrtoint ptr %c to i64
  %result = inttoptr i64 %d to ptr
  ret ptr %result
}

; --------------------------------------------------------------------------
; Functions that call other functions that either have pointer arguments, or
; return pointers.
; --------------------------------------------------------------------------

; CHECK-LABEL: define i64 @aaa
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = call ptr @aa(i64 %arg)
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 13
; CHECK-NEXT: [[PTRTOINT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[PTRTOINT]]
define i64 @aaa(i64 %arg) {
  %ptr = call ptr @aa(i64 %arg)
  %intptr = ptrtoint ptr %ptr to i64
  %with_offset = add i64 %intptr, 13
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @bbb
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 17
; CHECK-NEXT: [[CALL:%[a-zA-Z0-9_]+]] = call i64 @b(ptr [[GEP]])
; CHECK-NEXT: ret i64 [[CALL]]
define i64 @bbb(i64 %arg) {
  %with_offset = add i64 %arg, 17
  %ptr = inttoptr i64 %with_offset to ptr
  %result = call i64 @b(ptr %ptr)
  ret i64 %result
}

; =============================================================================
; =============================================================================
; Tests to check the transformation can handle different kinds of arithmetic,
; partly expressed as GEPs that are already present.
; =============================================================================
; =============================================================================

; Pointer argument, incremented via GEP already present.
; Then cast to integer and more arithmetic done via integer.
; Then casted back to pointer and returned.
;
; CHECK-LABEL: define ptr @c
; CHECK: [[ORIGINAL_GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %arg, i64 5
; CHECK-NEXT: [[NEW_GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 6
; CHECK-NEXT: ret ptr [[NEW_GEP]]
define ptr @c (ptr %arg) {
  %gep = getelementptr i8, ptr %arg, i64 5
  %intptr = ptrtoint ptr %gep to i64
  %with_offset = add i64 %intptr, 6
  %ptr_result = inttoptr i64 %with_offset to ptr
  ret ptr %ptr_result
}

; Pointer argument, cast to integer.
; Two consecutive adds turned into 2 GEPs.
; Then casted back to pointer and returned.
;
; CHECK-LABEL: define ptr @d
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr %arg, i64 7
; CHECK-NEXT: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 8
; CHECK-NEXT: ret ptr [[GEP2]]
define ptr @d (ptr %arg) {
  %intptr = ptrtoint ptr %arg to i64
  %plus7 = add i64 %intptr, 7
  %plus8 = add i64 %plus7, 8
  %ptr_result = inttoptr i64 %plus8 to ptr
  ret ptr %ptr_result
}

; =============================================================================
; =============================================================================
; Tests to check that the !revng.pointers metadata is used properly in various
; places.
; These tests basically mimic all the above, but instead of using ptr types, we
; we use integer types everywhere and then just use the !revng.pointers metadata
; to mark as pointers.
; =============================================================================
; =============================================================================

; ----------------------------------------------------
; Functions with pointer argument, integer return type
; ----------------------------------------------------

; Pointer argument, cast to integer, then incremented and returned as integer
;
; CHECK-LABEL: define i64 @xa
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 1
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xa (i64 %arg) !revng.pointers !1001 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

; Same as above, but with more casts back and forth between integer and pointer
;
; CHECK-LABEL: define i64 @xb
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 2
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xb (i64 %arg) !revng.pointers !1001 {
  %a = inttoptr i64 %arg to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  %c = inttoptr i64 %with_offset to ptr
  %d = ptrtoint ptr %c to i64
  ret i64 %d
}

; ----------------------------------------------------
; Functions with integer argument, pointer return type
; ----------------------------------------------------

; Pointer argument, cast to integer, then incremented and returned as integer
;
; CHECK-LABEL: define i64 @xaa
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 1
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xaa (i64 %arg) !revng.pointers !1002 {
  %with_offset = add i64 %arg, 1
  ret i64 %with_offset
}

; Same as above, but with more casts back and forth between integer and pointer
;
; CHECK-LABEL: define i64 @xbb
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 2
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xbb (i64 %arg) !revng.pointers !1002 {
  %a = inttoptr i64 %arg to ptr
  %b = ptrtoint ptr %a to i64
  %with_offset = add i64 %b, 2
  %c = inttoptr i64 %with_offset to ptr
  %d = ptrtoint ptr %c to i64
  ret i64 %d
}

; --------------------------------------------------------------------------
; Functions that call other functions that either have pointer arguments, or
; return pointers.
; --------------------------------------------------------------------------

; CHECK-LABEL: define i64 @xaaa
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = call i64 @xaa(i64 %arg)
; CHECK-NEXT: [[INTTOPTR:%[a-zA-Z0-9_]+]] = inttoptr i64 [[PTR]] to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[INTTOPTR]], i64 13
; CHECK-NEXT: [[PTRTOINT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[PTRTOINT]]
define i64 @xaaa(i64 %arg) !revng.pointers !1000 {
  %ptr = call i64 @xaa(i64 %arg)
  %with_offset = add i64 %ptr, 13
  ret i64 %with_offset
}

; CHECK-LABEL: define i64 @xbbb
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 17
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: [[CALL:%[a-zA-Z0-9_]+]] = call i64 @xb(i64 [[INT]])
; CHECK-NEXT: ret i64 [[CALL]]
define i64 @xbbb(i64 %arg) !revng.pointers !1000 {
  %with_offset = add i64 %arg, 17
  %result = call i64 @xb(i64 %with_offset)
  ret i64 %result
}

; ------------------------------------------------------------------------------
; Functions that call other functions indirectly, with call sites decorated with
; !revng.pointers metadata
; ------------------------------------------------------------------------------

; CHECK-LABEL: define i64 @indirect_a
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 31
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: [[CALL:%[a-zA-Z0-9_]+]] = call i64 undef(i64 [[INT]])
; CHECK-NEXT: ret i64 [[CALL]]
define i64 @indirect_a(i64 %arg) !revng.pointers !1000 {
  %with_offset = add i64 %arg, 31
  %result = call i64 undef(i64 %with_offset), !revng.pointers !1001
  ret i64 %result
}

; CHECK-LABEL: define i64 @indirect_b
; CHECK: [[CALL:%[a-zA-Z0-9_]+]] = call i64 undef(i64 %arg)
; CHECK-NEXT: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 [[CALL]] to ptr
; CHECK-NEXT: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 39
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @indirect_b(i64 %arg) !revng.pointers !1000 {
  %result = call i64 undef(i64 %arg), !revng.pointers !1002
  %with_offset = add i64 %result, 39
  ret i64 %with_offset
}

; =============================================================================
; =============================================================================
; Tests to check the transformation can handle different kinds of arithmetic,
; partly expressed as GEPs that are already present.
; =============================================================================
; =============================================================================

; Pointer argument, incremented via GEP already present.
; Then cast to integer and more arithmetic done via integer.
; Then casted back to pointer and returned.
;
; CHECK-LABEL: define i64 @xc
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[ORIGINAL_GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 5
; CHECK-NEXT: [[NEW_GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[ORIGINAL_GEP]], i64 6
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[NEW_GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xc (i64 %arg) !revng.pointers !1003 {
  %ptr = inttoptr i64 %arg to ptr
  %gep = getelementptr i8, ptr %ptr, i64 5
  %intptr = ptrtoint ptr %gep to i64
  %with_offset = add i64 %intptr, 6
  ret i64 %with_offset
}

; Pointer argument, cast to integer.
; Two consecutive adds turned into 2 GEPs.
; Then casted back to pointer and returned.
;
; CHECK-LABEL: define i64 @xd
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK-NEXT: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 7
; CHECK-NEXT: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 8
; CHECK-NEXT: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP2]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @xd (i64 %arg) !revng.pointers !1003 {
  %plus7 = add i64 %arg, 7
  %plus8 = add i64 %plus7, 8
  ret i64 %plus8
}

; ----------------------------------------------------------------------
; Check that if !revng.pointers is missing we never promote adds to GEPs
; ----------------------------------------------------------------------

; CHECK-LABEL: define i64 @dont_optimize
; CHECK: [[ADD:%[a-zA-Z0-9_]+]] = add i64 %arg, 47
; CHECK-NEXT: ret i64 [[ADD]]
define i64 @dont_optimize(i64 %arg) {
  %result = add i64 %arg, 47
  ret i64 %result
}

!0 = !{ i1 false }
!1 = !{ i1 true }
; non-pointer return return type, non-pointer operand type
!1000 = !{ !0, !0 }
; non-pointer return return type, pointer operand type
!1001 = !{ !0, !1 }
; pointer return return type, non-pointer operand type
!1002 = !{ !1, !0 }
; pointer return return type, pointer operand type
!1003 = !{ !1, !1 }
