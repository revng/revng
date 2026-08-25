;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -split-exponential-dataflow -split-exponential-dataflow-threshold=8 -S -o - | FileCheck %s

; The threshold is the largest expression, in nodes, the clifter may build.
; Leaves cost one node: arguments, constants, globals, and anything this pass
; has already moved into a local variable.

@global = external global i64

declare void @side_effect(i64)
declare { i64, i64 } @returns_struct(i64)

declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; =============================================================================
; An expression that stays within the threshold is left alone.
; Sizes: %sum 3, %prod 5, %mixed 7.
; =============================================================================

define i64 @under_threshold(i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  %prod = mul i64 %sum, 3
  %mixed = xor i64 %prod, 7
  ret i64 %mixed
}

; CHECK-LABEL: define i64 @under_threshold
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %sum = add i64 %a, %b
; CHECK-NEXT:    %prod = mul i64 %sum, 3
; CHECK-NEXT:    %mixed = xor i64 %prod, 7
; CHECK-NEXT:    ret i64 %mixed

; =============================================================================
; One node past the threshold is enough, and only one variable is needed:
; counting restarts from the split.
; Sizes: %sum 3, %prod 5, %mixed 7, %big 9 -> split, %result 3.
; =============================================================================

define i64 @over_threshold(i64 %a, i64 %b) {
entry:
  %sum = add i64 %a, %b
  %prod = mul i64 %sum, 3
  %mixed = xor i64 %prod, 7
  %big = sub i64 %mixed, 1
  %result = add i64 %big, 2
  ret i64 %result
}

; CHECK-LABEL: define i64 @over_threshold
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VAR:%[0-9]+]] = alloca i64
; CHECK-NEXT:    %sum = add i64 %a, %b
; CHECK-NEXT:    %prod = mul i64 %sum, 3
; CHECK-NEXT:    %mixed = xor i64 %prod, 7
; CHECK-NEXT:    %big = sub i64 %mixed, 1
; CHECK-NEXT:    store i64 %big, ptr [[VAR]]
; CHECK-NEXT:    [[COPY:%[0-9]+]] = load i64, ptr [[VAR]]
; CHECK-NEXT:    %result = add i64 [[COPY]], 2
; CHECK-NEXT:    ret i64 %result

; =============================================================================
; The shape that motivates the pass: each round reads the previous round's
; accumulator twice, doubling the expression every round. Splitting keeps it
; linear, one variable per round.
; =============================================================================

define i64 @diamond(i64 %a) {
entry:
  %r0 = add i64 %a, 1
  %r0.lo = shl i64 %r0, 2
  %r0.hi = lshr i64 %r0, 3
  %r1 = xor i64 %r0.lo, %r0.hi
  %r1.lo = shl i64 %r1, 2
  %r1.hi = lshr i64 %r1, 3
  %r1.x = xor i64 %r1.lo, %r1.hi
  %r2 = xor i64 %r1.x, %a
  ret i64 %r2
}

; CHECK-LABEL: define i64 @diamond
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VAR0:%[0-9]+]] = alloca i64
; CHECK-NEXT:    [[VAR1:%[0-9]+]] = alloca i64
; CHECK-NEXT:    %r0 = add i64 %a, 1
; CHECK-NEXT:    %r0.lo = shl i64 %r0, 2
; CHECK-NEXT:    %r0.hi = lshr i64 %r0, 3
; CHECK-NEXT:    %r1 = xor i64 %r0.lo, %r0.hi
; CHECK-NEXT:    store i64 %r1, ptr [[VAR0]]
; CHECK-NEXT:    [[COPY0:%[0-9]+]] = load i64, ptr [[VAR0]]
; CHECK-NEXT:    %r1.lo = shl i64 [[COPY0]], 2
; CHECK-NEXT:    %r1.hi = lshr i64 [[COPY0]], 3
; CHECK-NEXT:    %r1.x = xor i64 %r1.lo, %r1.hi
; CHECK-NEXT:    %r2 = xor i64 %r1.x, %a
; CHECK-NEXT:    store i64 %r2, ptr [[VAR1]]
; CHECK-NEXT:    [[COPY1:%[0-9]+]] = load i64, ptr [[VAR1]]
; CHECK-NEXT:    ret i64 [[COPY1]]

; =============================================================================
; Leaves cost one node each, so a wide but shallow expression stays inline.
; Sizes: %x 3, %y 3, %z 7.
; =============================================================================

define i64 @leaves_are_cheap(i64 %a, i64 %b, i64 %c) {
entry:
  %x = add i64 %a, ptrtoint (ptr @global to i64)
  %y = add i64 %b, %c
  %z = add i64 %x, %y
  ret i64 %z
}

; CHECK-LABEL: define i64 @leaves_are_cheap
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %x = add i64 %a, ptrtoint (ptr @global to i64)
; CHECK-NEXT:    %y = add i64 %b, %c
; CHECK-NEXT:    %z = add i64 %x, %y
; CHECK-NEXT:    ret i64 %z

; =============================================================================
; Allocas, stores and calls with no uses are already statements, and are
; never moved into a variable.
; =============================================================================

define void @statements_are_never_split(i64 %a, ptr %p) {
entry:
  %local = alloca i64
  %t1 = add i64 %a, 1
  %t2 = mul i64 %t1, 3
  store i64 %t2, ptr %p
  store i64 %t2, ptr %local
  call void @side_effect(i64 %t2)
  ret void
}

; CHECK-LABEL: define void @statements_are_never_split
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %local = alloca i64
; CHECK-NEXT:    %t1 = add i64 %a, 1
; CHECK-NEXT:    %t2 = mul i64 %t1, 3
; CHECK-NEXT:    store i64 %t2, ptr %p
; CHECK-NEXT:    store i64 %t2, ptr %local
; CHECK-NEXT:    call void @side_effect(i64 %t2)
; CHECK-NEXT:    ret void

; =============================================================================
; A new variable goes after the allocas already in the entry block.
; =============================================================================

define i64 @split_after_existing_allocas(i64 %a, ptr %p) {
entry:
  %local = alloca i64
  %t1 = add i64 %a, 1
  %t2 = mul i64 %t1, 3
  %t3 = xor i64 %t2, 7
  %t4 = sub i64 %t3, 1
  store i64 %t4, ptr %local
  ret i64 %t4
}

; CHECK-LABEL: define i64 @split_after_existing_allocas
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %local = alloca i64
; CHECK-NEXT:    [[VAR:%[0-9]+]] = alloca i64
; CHECK-NEXT:    %t1 = add i64 %a, 1
; CHECK-NEXT:    %t2 = mul i64 %t1, 3
; CHECK-NEXT:    %t3 = xor i64 %t2, 7
; CHECK-NEXT:    %t4 = sub i64 %t3, 1
; CHECK-NEXT:    store i64 %t4, ptr [[VAR]]
; CHECK-NEXT:    [[COPY:%[0-9]+]] = load i64, ptr [[VAR]]
; CHECK-NEXT:    store i64 [[COPY]], ptr %local
; CHECK-NEXT:    ret i64 [[COPY]]

; =============================================================================
; An aggregate is not special: it goes into a local variable like anything
; else, allocated as a byte array of the same size.
; Sizes: %t1 3, %t2 5, %t3 7, %agg 9 -> split.
; =============================================================================

define i64 @aggregates_are_not_special(i64 %a) {
entry:
  %t1 = add i64 %a, 1
  %t2 = mul i64 %t1, 3
  %t3 = xor i64 %t2, 7
  %agg = call { i64, i64 } @returns_struct(i64 %t3)
  %lo = extractvalue { i64, i64 } %agg, 0
  ret i64 %lo
}

; CHECK-LABEL: define i64 @aggregates_are_not_special
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VAR:%[0-9]+]] = alloca [16 x i8]
; CHECK-NEXT:    %t1 = add i64 %a, 1
; CHECK-NEXT:    %t2 = mul i64 %t1, 3
; CHECK-NEXT:    %t3 = xor i64 %t2, 7
; CHECK-NEXT:    %agg = call { i64, i64 } @returns_struct(i64 %t3)
; CHECK-NEXT:    store { i64, i64 } %agg, ptr [[VAR]]
; CHECK-NEXT:    [[COPY:%[0-9]+]] = load { i64, i64 }, ptr [[VAR]]
; CHECK-NEXT:    %lo = extractvalue { i64, i64 } [[COPY]], 0
; CHECK-NEXT:    ret i64 %lo

; =============================================================================
; Scope-graph markers are not emitted: they contribute nothing to any
; expression and are left untouched.
; =============================================================================

define void @markers_are_ignored(i64 %a, ptr %p) {
entry:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@markers_are_ignored, %other))
  %t1 = add i64 %a, 1
  store i64 %t1, ptr %p
  br label %other

other:
  ret void
}

; CHECK-LABEL: define void @markers_are_ignored
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @goto_block()
; CHECK-NEXT:    call void @scope_closer(ptr blockaddress(@markers_are_ignored, %other))
; CHECK-NEXT:    %t1 = add i64 %a, 1
; CHECK-NEXT:    store i64 %t1, ptr %p
; CHECK-NEXT:    br label %other
; CHECK:       other:
; CHECK-NEXT:    ret void
