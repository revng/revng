;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -simplify-comparisons -early-type-shrinking -type-shrinking -early-cse -instcombine -early-type-shrinking -type-shrinking -early-cse %s -S -o - | FileCheck %s

; Recover signed comparisons from packed sign and zero flags.
; InstCombine joins the predicates after type shrinking makes their widths agree.

; CHECK-LABEL: define i1 @positive_i32(i64 %x)
define i1 @positive_i32(i64 %x) {
entry:
  %low = and i64 %x, 4294967295
  %iszero = icmp eq i64 %low, 0
  %zf = select i1 %iszero, i64 64, i64 0
  %shifted = lshr i64 %x, 24
  %sf = and i64 %shifted, 128
  %flags = or i64 %zf, %sf
  %positive = icmp eq i64 %flags, 0
  ; CHECK: [[VALUE:%[a-zA-Z0-9._]+]] = trunc i64 %x to i32
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = icmp sgt i32 [[VALUE]], 0
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %positive
}

; CHECK-LABEL: define i1 @nonpositive_i16(i32 %x)
define i1 @nonpositive_i16(i32 %x) {
entry:
  %low = and i32 %x, 65535
  %iszero = icmp eq i32 %low, 0
  %zf = select i1 %iszero, i32 1, i32 0
  %shifted = ashr i32 %x, 8
  %sf = and i32 %shifted, 128
  %flags = or i32 %sf, %zf
  %nonpositive = icmp ne i32 %flags, 0
  ; CHECK: [[VALUE:%[a-zA-Z0-9._]+]] = trunc i32 %x to i16
  ; CHECK-NEXT: [[RESULT:%[a-zA-Z0-9._]+]] = icmp slt i16 [[VALUE]], 1
  ; CHECK-NEXT: ret i1 [[RESULT]]
  ret i1 %nonpositive
}
