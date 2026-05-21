;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -arithmetic-to-gep -S -o - | FileCheck %s

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

; =============================================================================
; =============================================================================
; Tests with deep cones of dataflow, where many add instructions are chained
; and only one of the operands is a valid base pointer. The base pointer is
; not used directly from the root AddInst, so discovery has to propagate
; through many layers of ambiguous Add operations to find it.
; =============================================================================
; =============================================================================

; Eight chained adds. The non-pointer operand of every add is the result of a
; clearly-non-pointer operation (mul, ashr, shl, and, or, xor, sdiv, srem).
; Discovery starts from the !revng.pointers metadata on the function (return
; type is a pointer) and must walk backwards through 8 adds before it lands
; on %arg as the only viable base pointer.
;
; CHECK-LABEL: define i64 @deep_chain_to_return
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %as
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %sh
; CHECK: [[GEP4:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP3]], i64 %an
; CHECK: [[GEP5:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP4]], i64 %o1
; CHECK: [[GEP6:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP5]], i64 %x1
; CHECK: [[GEP7:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP6]], i64 %dv
; CHECK: [[GEP8:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP7]], i64 %rm
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP8]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @deep_chain_to_return(i64 %arg, i64 %a, i64 %b) !revng.pointers !2002 {
  %m  = mul  i64 %a, %b
  %a1 = add  i64 %arg, %m
  %as = ashr i64 %a, 3
  %a2 = add  i64 %a1, %as
  %sh = shl  i64 %a, 4
  %a3 = add  i64 %a2, %sh
  %an = and  i64 %a, %b
  %a4 = add  i64 %a3, %an
  %o1 = or   i64 %a, %b
  %a5 = add  i64 %a4, %o1
  %x1 = xor  i64 %a, %b
  %a6 = add  i64 %a5, %x1
  %dv = sdiv i64 %a, %b
  %a7 = add  i64 %a6, %dv
  %rm = srem i64 %a, %b
  %a8 = add  i64 %a7, %rm
  ret i64 %a8
}

; Pointer argument drives a chain of 4 adds. The non-pointer operand of every
; add is a clearly-non-pointer multiplication. Here there is no backwards
; discovery: the obvious pointer is %arg itself. The chain is rewritten by
; propagating forward through each Add user.
;
; CHECK-LABEL: define i64 @deep_chain_from_arg
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m1
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %m2
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %m3
; CHECK: [[GEP4:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP3]], i64 %m4
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP4]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @deep_chain_from_arg(i64 %arg, i64 %a, i64 %b) !revng.pointers !2001 {
  %m1 = mul i64 %a, %b
  %a1 = add i64 %arg, %m1
  %m2 = mul i64 %a, %b
  %a2 = add i64 %a1, %m2
  %m3 = mul i64 %a, %b
  %a3 = add i64 %a2, %m3
  %m4 = mul i64 %a, %b
  %a4 = add i64 %a3, %m4
  ret i64 %a4
}

; A chain of 4 adds feeds the integer argument of a call whose argument is
; known (via metadata) to be a pointer. Discovery starts from the call-site
; pointer use and walks backwards through 4 adds to identify %arg as the
; base pointer.
;
; CHECK-LABEL: define i64 @deep_chain_to_call_arg
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m1
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %m2
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %m3
; CHECK: [[GEP4:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP3]], i64 %m4
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP4]] to i64
; CHECK: [[CALL:%[a-zA-Z0-9_]+]] = call i64 @xb(i64 [[INT]])
; CHECK-NEXT: ret i64 [[CALL]]
define i64 @deep_chain_to_call_arg(i64 %arg, i64 %a, i64 %b) !revng.pointers !2000 {
  %m1 = mul i64 %a, %b
  %a1 = add i64 %arg, %m1
  %m2 = mul i64 %a, %b
  %a2 = add i64 %a1, %m2
  %m3 = mul i64 %a, %b
  %a3 = add i64 %a2, %m3
  %m4 = mul i64 %a, %b
  %a4 = add i64 %a3, %m4
  %result = call i64 @xb(i64 %a4)
  ret i64 %result
}

; Chain of 4 adds where the position of the pointer operand alternates
; between operand 0 and operand 1 of each add. Discovery must check both
; sides of every add when disambiguating.
;
; CHECK-LABEL: define i64 @mixed_operand_order
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m1
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %m2
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %m3
; CHECK: [[GEP4:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP3]], i64 %m4
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP4]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @mixed_operand_order(i64 %arg, i64 %a, i64 %b) !revng.pointers !2002 {
  %m1 = mul i64 %a, %b
  %a1 = add i64 %arg, %m1   ; pointer is operand 0
  %m2 = mul i64 %a, %b
  %a2 = add i64 %m2, %a1    ; pointer is operand 1
  %m3 = mul i64 %a, %b
  %a3 = add i64 %a2, %m3    ; pointer is operand 0
  %m4 = mul i64 %a, %b
  %a4 = add i64 %m4, %a3    ; pointer is operand 1
  ret i64 %a4
}

; Chain of 2 adds where the non-pointer operand of each add is reached
; through a freeze of a multiplication. cannotBePointer must recurse through
; freeze to determine that the operand cannot be a pointer.
;
; CHECK-LABEL: define i64 @non_pointer_via_freeze
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %f1
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %f2
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP2]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @non_pointer_via_freeze(i64 %arg, i64 %a, i64 %b) !revng.pointers !2002 {
  %m1 = mul i64 %a, %b
  %f1 = freeze i64 %m1
  %a1 = add i64 %arg, %f1
  %m2 = mul i64 %a, %b
  %f2 = freeze i64 %m2
  %a2 = add i64 %a1, %f2
  ret i64 %a2
}

; Chain of 3 adds that all reuse the same multiplication as their offset.
; This exercises the fact that disambiguation is done independently for each
; add and is robust to the offset side being a shared sub-expression.
;
; CHECK-LABEL: define i64 @shared_offset_mul
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %m
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %m
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP3]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @shared_offset_mul(i64 %arg, i64 %a, i64 %b) !revng.pointers !2002 {
  %m  = mul i64 %a, %b
  %a1 = add i64 %arg, %m
  %a2 = add i64 %a1, %m
  %a3 = add i64 %a2, %m
  ret i64 %a3
}

; Chain where the chain itself is interrupted by ptrtoint/inttoptr round
; trips between adds. cannotBePointer's recursion through these casts (via
; the same paths used for transparent passthroughs) must still allow
; disambiguation to identify the chain side.
;
; CHECK-LABEL: define i64 @chain_with_intermediate_casts
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP1:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 %m1
; CHECK: [[GEP2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP1]], i64 %m2
; CHECK: [[GEP3:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[GEP2]], i64 %m3
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP3]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @chain_with_intermediate_casts(i64 %arg, i64 %a, i64 %b) !revng.pointers !2002 {
  %m1 = mul i64 %a, %b
  %a1 = add i64 %arg, %m1
  %p1 = inttoptr i64 %a1 to ptr
  %i1 = ptrtoint ptr %p1 to i64
  %m2 = mul i64 %a, %b
  %a2 = add i64 %i1, %m2
  %p2 = inttoptr i64 %a2 to ptr
  %i2 = ptrtoint ptr %p2 to i64
  %m3 = mul i64 %a, %b
  %a3 = add i64 %i2, %m3
  ret i64 %a3
}

; =============================================================================
; =============================================================================
; Tests where the only initial pointer is the return value (per the
; !revng.pointers metadata) and is computed as
;
;   %arg + (%a * %b + %c * %d)   (and deeper variants)
;
; %arg is the actual base pointer but is not used directly from the root add:
; the offset side is itself an add of all-non-pointer multiplications. To
; disambiguate the root add, the discovery walk has to recognise that the
; inner add of two clearly-non-pointer values is itself a non-pointer.
;
; This means cannotBePointer must recurse through Add when both operands
; cannot be pointers.
; =============================================================================
; =============================================================================

; The simplest case: %result = %arg + (%a*%b + %c*%d). Discovery from the
; pointer return must see that the inner add (%a*%b + %c*%d) cannot be a
; pointer because both its operands are clearly non-pointer multiplications,
; and conclude that %arg is the only viable base pointer of the outer add.
;
; CHECK-LABEL: define i64 @backward_through_inner_add_subtree
; CHECK: [[M1:%[a-zA-Z0-9_]+]] = mul i64 %a, %b
; CHECK: [[M2:%[a-zA-Z0-9_]+]] = mul i64 %c, %d
; CHECK: [[OFFSET:%[a-zA-Z0-9_]+]] = add i64 [[M1]], [[M2]]
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 [[OFFSET]]
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @backward_through_inner_add_subtree(i64 %arg,
                                                i64 %a, i64 %b,
                                                i64 %c, i64 %d)
                                                !revng.pointers !2102 {
  %m1 = mul i64 %a, %b
  %m2 = mul i64 %c, %d
  %inner = add i64 %m1, %m2
  %result = add i64 %arg, %inner
  ret i64 %result
}

; Same idea but with a deeper non-pointer cone: the offset is
; ((%a*%b + %c*%d) + (%e*%f + %g*%h)). cannotBePointer must recurse through
; two layers of Add to determine that the whole offset cannot be a pointer.
;
; CHECK-LABEL: define i64 @backward_through_deep_inner_add_subtree
; CHECK: [[M1:%[a-zA-Z0-9_]+]] = mul i64 %a, %b
; CHECK: [[M2:%[a-zA-Z0-9_]+]] = mul i64 %c, %d
; CHECK: [[M3:%[a-zA-Z0-9_]+]] = mul i64 %e, %f
; CHECK: [[M4:%[a-zA-Z0-9_]+]] = mul i64 %g, %h
; CHECK: [[I1:%[a-zA-Z0-9_]+]] = add i64 [[M1]], [[M2]]
; CHECK: [[I2:%[a-zA-Z0-9_]+]] = add i64 [[M3]], [[M4]]
; CHECK: [[OFFSET:%[a-zA-Z0-9_]+]] = add i64 [[I1]], [[I2]]
; CHECK: [[PTR:%[a-zA-Z0-9_]+]] = inttoptr i64 %arg to ptr
; CHECK: [[GEP:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr [[PTR]], i64 [[OFFSET]]
; CHECK: [[INT:%[a-zA-Z0-9_]+]] = ptrtoint ptr [[GEP]] to i64
; CHECK-NEXT: ret i64 [[INT]]
define i64 @backward_through_deep_inner_add_subtree(i64 %arg,
                                                     i64 %a, i64 %b,
                                                     i64 %c, i64 %d,
                                                     i64 %e, i64 %f,
                                                     i64 %g, i64 %h)
                                                     !revng.pointers !2202 {
  %m1 = mul i64 %a, %b
  %m2 = mul i64 %c, %d
  %m3 = mul i64 %e, %f
  %m4 = mul i64 %g, %h
  %inner1 = add i64 %m1, %m2
  %inner2 = add i64 %m3, %m4
  %offset = add i64 %inner1, %inner2
  %result = add i64 %arg, %offset
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

; Metadata used by the deep-chain tests below, which all take 3 i64 arguments
; (only the first one being a pointer, when applicable).
!2 = !{ i1 false, i1 false, i1 false }
!3 = !{ i1 true, i1 false, i1 false }
; 3-args: non-pointer return, no pointer operand
!2000 = !{ !0, !2 }
; 3-args: non-pointer return, first operand is a pointer
!2001 = !{ !0, !3 }
; 3-args: pointer return, no pointer operand
!2002 = !{ !1, !2 }

; Metadata for the 5- and 9-argument backward-discovery tests below. None of
; the args are flagged as pointers; only the return value is.
!4 = !{ i1 false, i1 false, i1 false, i1 false, i1 false }
!5 = !{ i1 false, i1 false, i1 false, i1 false, i1 false,
        i1 false, i1 false, i1 false, i1 false }
; 5-args: pointer return, no pointer operand
!2102 = !{ !1, !4 }
; 9-args: pointer return, no pointer operand
!2202 = !{ !1, !5 }
