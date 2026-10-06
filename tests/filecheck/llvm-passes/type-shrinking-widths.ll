;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt -S -type-shrinking -early-cse -dce -verify %s | FileCheck --check-prefixes=CHECK,DIRECT %s
; RUN: %root/bin/revng opt -S -type-shrinking -instcombine -type-shrinking -early-cse -dce -verify %s | FileCheck %s
; RUN: %root/bin/revng opt -S -type-shrinking -verify %s | FileCheck --check-prefix=RAW %s
; RUN: %root/bin/revng opt -S -type-shrinking -min-width=16 -early-cse -dce -verify %s | FileCheck --check-prefix=MIN %s

; All result bits are observed, but the variable mask bounds the value.
define i64 @variable_mask(i64 %x, i8 %mask) {
  ; CHECK-LABEL: @variable_mask(
  ; CHECK: and i8
  ; CHECK: zext i8 {{.*}} to i64
  ; MIN-LABEL: @variable_mask(
  ; MIN: and i16
  %m = zext i8 %mask to i64
  %r = and i64 %x, %m
  ret i64 %r
}

; Forward bounds preserve the carry when the whole sum is returned.
define i64 @unsigned_sum(i8 %x, i8 %y) {
  ; CHECK-LABEL: @unsigned_sum(
  ; CHECK: add i16
  ; CHECK: zext i16 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  %r = add i64 %a, %b
  ret i64 %r
}

define i64 @unsigned_difference(i8 %x, i8 %y) {
  ; CHECK-LABEL: @unsigned_difference(
  ; CHECK: sub i16
  ; CHECK: sext i16 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  %r = sub i64 %a, %b
  ret i64 %r
}

; The unknown sign bit is replicated, rather than individually known.
define i64 @signed_bits(i8 %x, i8 %y) {
  ; CHECK-LABEL: @signed_bits(
  ; CHECK: xor i8
  ; CHECK: sext i8 {{.*}} to i64
  %a = sext i8 %x to i64
  %b = sext i8 %y to i64
  %r = xor i64 %a, %b
  ret i64 %r
}

; Demand can narrow an operation whose full value has no narrow bound.
define i8 @demanded_sum(i64 %x, i64 %y) {
  ; CHECK-LABEL: @demanded_sum(
  ; CHECK: add i8
  ; CHECK: ret i8
  %r = add i64 %x, %y
  %low = trunc i64 %r to i8
  ret i8 %low
}

; Join zero and sign extension in a representation that preserves both ranges.
define i64 @mixed_select(i1 %c, i8 %x, i8 %y) {
  ; CHECK-LABEL: @mixed_select(
  ; CHECK: select i1 %c, i16
  ; CHECK: sext i16 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = sext i8 %y to i64
  %r = select i1 %c, i64 %a, i64 %b
  ret i64 %r
}

define i64 @signed_select(i1 %c, i8 %x) {
  ; CHECK-LABEL: @signed_select(
  ; CHECK: select i1 %c, i8
  ; CHECK: sext i8 {{.*}} to i64
  %a = sext i8 %x to i64
  %r = select i1 %c, i64 %a, i64 -42
  ret i64 %r
}

define i64 @different_widths(i1 %c, i8 %x, i16 %y) {
  ; CHECK-LABEL: @different_widths(
  ; CHECK: select i1 %c, i16
  ; CHECK: zext i16 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = zext i16 %y to i64
  %r = select i1 %c, i64 %a, i64 %b
  ret i64 %r
}

; Arbitrary incoming values must prevent exact narrowing of the merge.
define i64 @unknown_incoming(i1 %c, i8 %x, i64 %y) {
  ; CHECK-LABEL: @unknown_incoming(
  ; CHECK: select i1 %c, i64
  %a = zext i8 %x to i64
  %r = select i1 %c, i64 %a, i64 %y
  ret i64 %r
}

; Freezing a possibly poisoned extension can produce any wide value.
define i64 @freeze_boundary(i1 %c, i8 %x, i8 %y) {
  ; CHECK-LABEL: @freeze_boundary(
  ; DIRECT: freeze i64
  ; DIRECT: select i1 %c, i64
  %a = zext i8 %x to i64
  %f = freeze i64 %a
  %b = zext i8 %y to i64
  %r = select i1 %c, i64 %f, i64 %b
  ret i64 %r
}

; The common representation is inferred through arithmetic, not matched casts.
define i1 @compare_computations(i8 %x, i8 %y) {
  ; CHECK-LABEL: @compare_computations(
  ; CHECK: add i16
  ; CHECK: icmp u{{lt|gt}} i16
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  %sum = add i64 %a, %b
  %r = icmp slt i64 %sum, 300
  ret i1 %r
}

define i1 @compare_mixed_extensions(i8 %x, i8 %y) {
  ; CHECK-LABEL: @compare_mixed_extensions(
  ; CHECK: icmp slt i16
  %a = zext i8 %x to i64
  %b = sext i8 %y to i64
  %r = icmp slt i64 %a, %b
  ret i1 %r
}

define i64 @signed_shift(i8 %x) {
  ; CHECK-LABEL: @signed_shift(
  ; CHECK: ashr i8 {{.*}}, 3
  ; CHECK: sext i8 {{.*}} to i64
  %a = sext i8 %x to i64
  %r = ashr i64 %a, 3
  ret i64 %r
}

define i64 @bounded_shift(i8 %x, i8 %amount) {
  ; CHECK-LABEL: @bounded_shift(
  ; CHECK: lshr i8
  ; CHECK: zext i8 {{.*}} to i64
  %a = zext i8 %x to i64
  %masked = and i8 %amount, 7
  %n = zext i8 %masked to i64
  %r = lshr i64 %a, %n
  ret i64 %r
}

; Eight is a defined wide shift count: do not turn it into an i8 overshift.
define i64 @shift_boundary(i8 %x, i8 %amount) {
  ; CHECK-LABEL: @shift_boundary(
  ; CHECK: lshr i16
  %a = zext i8 %x to i64
  %masked = and i8 %amount, 15
  %n = zext i8 %masked to i64
  %r = lshr i64 %a, %n
  ret i64 %r
}

define i64 @unknown_shift(i8 %x, i64 %amount) {
  ; CHECK-LABEL: @unknown_shift(
  ; CHECK: lshr i64
  %a = zext i8 %x to i64
  %r = lshr i64 %a, %amount
  ret i64 %r
}

; Low result bits do not suffice to truncate division operands.
define i8 @wide_division(i64 %x, i64 %y) {
  ; CHECK-LABEL: @wide_division(
  ; CHECK: udiv i64
  %q = udiv i64 %x, %y
  %r = trunc i64 %q to i8
  ret i8 %r
}

define i64 @unsigned_division(i8 %x, i8 %y) {
  ; CHECK-LABEL: @unsigned_division(
  ; CHECK: udiv i8
  ; CHECK: zext i8 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  %r = udiv i64 %a, %b
  ret i64 %r
}

; Both sdiv and srem must avoid introducing INT_MIN / -1 at the new width.
define i64 @signed_division(i8 %x, i8 %y) {
  ; CHECK-LABEL: @signed_division(
  ; CHECK: sdiv i16
  ; CHECK: sext i16 {{.*}} to i64
  %a = sext i8 %x to i64
  %b = sext i8 %y to i64
  %r = sdiv i64 %a, %b
  ret i64 %r
}

define i64 @signed_remainder(i8 %x, i8 %y) {
  ; CHECK-LABEL: @signed_remainder(
  ; CHECK: srem i16
  ; CHECK: trunc i16 {{.*}} to i8
  ; CHECK: sext i8 {{.*}} to i64
  %a = sext i8 %x to i64
  %b = sext i8 %y to i64
  %r = srem i64 %a, %b
  ret i64 %r
}

define i64 @safe_signed_division(i8 %x) {
  ; CHECK-LABEL: @safe_signed_division(
  ; CHECK: {{sdiv|ashr}} i8
  ; CHECK: sext i8 {{.*}} to i64
  %a = sext i8 %x to i64
  %r = sdiv i64 %a, 2
  ret i64 %r
}

; A shared producer must still supply all bits observed by its wide consumer.
define i8 @shared_wide_use(i64 %x, i64 %y, ptr %out) {
  ; CHECK-LABEL: @shared_wide_use(
  ; CHECK: add i64
  ; CHECK: store i64
  %sum = add i64 %x, %y
  store i64 %sum, ptr %out
  %r = trunc i64 %sum to i8
  ret i8 %r
}

; Dropping nowrap is required even if narrowing only changes its operands.
define i8 @nowrap(i64 %x, i64 %y) {
  ; CHECK-LABEL: @nowrap(
  ; CHECK: add i8
  %sum = add nuw nsw i64 %x, %y
  %r = trunc i64 %sum to i8
  ret i8 %r
}

; A genuine cycle through arithmetic and a mask has a stable narrow range.
define i64 @masked_recurrence(i8 %initial, i32 %n) {
  ; CHECK-LABEL: @masked_recurrence(
entry:
  %a = zext i8 %initial to i64
  br label %loop
loop:
  ; CHECK: phi i8
  %value = phi i64 [ %a, %entry ], [ %masked, %loop ]
  %index = phi i32 [ 0, %entry ], [ %nextindex, %loop ]
  %next = add i64 %value, 1
  %masked = and i64 %next, 255
  %nextindex = add i32 %index, 1
  %done = icmp eq i32 %nextindex, %n
  br i1 %done, label %exit, label %loop
exit:
  ; CHECK: zext i8 {{.*}} to i64
  ret i64 %masked
}

; Without the mask, the seed's width does not bound the recurrence.
define i64 @growing_recurrence(i8 %initial, i64 %n) {
  ; CHECK-LABEL: @growing_recurrence(
entry:
  %a = zext i8 %initial to i64
  br label %loop
loop:
  ; CHECK: phi i64
  %value = phi i64 [ %a, %entry ], [ %next, %loop ]
  %index = phi i64 [ 0, %entry ], [ %nextindex, %loop ]
  ; CHECK: add i64
  %next = add i64 %value, 1
  %nextindex = add i64 %index, 1
  %done = icmp eq i64 %nextindex, %n
  br i1 %done, label %exit, label %loop
exit:
  ret i64 %next
}

; Multiple PHIs in the same block, cyclic references, and a select backedge.
define i64 @cyclic_merges(i8 %x, i8 %y, i1 %choose, i1 %again) {
  ; CHECK-LABEL: @cyclic_merges(
entry:
  %a = sext i8 %x to i64
  %b = sext i8 %y to i64
  br label %loop
loop:
  ; CHECK: phi i8
  ; CHECK: phi i8
  %p = phi i64 [ %a, %entry ], [ %q, %loop ]
  %q = phi i64 [ %b, %entry ], [ %s, %loop ]
  %s = select i1 %choose, i64 %p, i64 %q
  br i1 %again, label %loop, label %exit
exit:
  ; CHECK: sext i8 {{.*}} to i64
  ret i64 %s
}

; Reuse exactly the same incoming cast for duplicate predecessor edges.
define i64 @duplicate_edges(i32 %which, i8 %x, i8 %y) {
  ; CHECK-LABEL: @duplicate_edges(
entry:
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  switch i32 %which, label %other [ i32 0, label %join
                                  i32 1, label %join ]
other:
  br label %join
join:
  ; CHECK: phi i8
  %r = phi i64 [ %a, %entry ], [ %a, %entry ], [ %b, %other ]
  ret i64 %r
}

; Neither an unseeded cycle nor a constant on a dead edge widens a live PHI.
define i64 @unreachable_incoming(i8 %x, i8 %y, i1 %c) {
  ; RAW-LABEL: @unreachable_incoming(
entry:
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  br i1 %c, label %live, label %join
live:
  br label %join
dead:
  ; RAW: %cycle = phi i64 [ %cycle, %dead ]
  %cycle = phi i64 [ %cycle, %dead ]
  br i1 %c, label %dead, label %join
dead_constant:
  br label %join
join:
  ; RAW: phi i8 [ %x, %entry ], [ %y, %live ], [ %{{[^ ]+}}, %dead ], [ -1, %dead_constant ]
  ; RAW: zext i8 {{.*}} to i64
  %r = phi i64 [ %a, %entry ], [ %b, %live ], [ %cycle, %dead ], [ -1, %dead_constant ]
  ret i64 %r
}

; SCCP's range for this select may include undef. Such a range provides no
; bound, so the select stays wide.
define i64 @undef_incoming(i1 %c, i8 %x) {
  ; RAW-LABEL: @undef_incoming(
  ; RAW: select i1 %c, i64
  %a = zext i8 %x to i64
  %r = select i1 %c, i64 %a, i64 undef
  ret i64 %r
}

; The unused division still executes, so it demands every bit of its divisor.
; Narrowing %d to the bits the live trunc demands would drop the nonzero bit
; and divide by zero, so %d stays wide and nothing is rewritten.
define i8 @unused_divisor(i64 %x) {
  ; RAW-LABEL: @unused_divisor(
  ; RAW-NOT: or i8
  ; RAW-NOT: udiv i8
  ; RAW-NOT: udiv i16
  ; RAW-NOT: udiv i32
  ; RAW: ret i8
  %d = or i64 %x, 256
  %unused = udiv i64 42, %d
  %r = trunc i64 %d to i8
  ret i8 %r
}

; A dead call chain keeps its producer wide, because every bit of the argument
; reaches a call this pass does not remove. DCE drops the chain afterwards and
; a later run then narrows %sum.
declare i64 @pure(i64 noundef) readnone nounwind willreturn
define i8 @unused_calls(i64 %x) {
  ; RAW-LABEL: @unused_calls(
  ; RAW-NOT: add i8
  ; RAW: ret i8
  %sum = add i64 %x, 1
  %unused = call i64 @pure(i64 %sum)
  %also_unused = call i64 @pure(i64 %unused)
  %r = trunc i64 %sum to i8
  ret i8 %r
}

; A dead cycle leaves an instruction with flags without a plan. It still
; receives the whole value of its operand, which only its bounds narrow, so its
; nsw flag stays valid.
define i32 @dead_flagged_user(i32 %x, i1 %continue) {
  ; RAW-LABEL: @dead_flagged_user(
  ; RAW: %[[MASKED:[^ ]+]] = and i8 {{.*}}, 3
  ; RAW: %[[WIDE:[^ ]+]] = zext i8 %[[MASKED]] to i32
  ; RAW-NEXT: sub nsw i32 0, %[[WIDE]]
entry:
  %masked = and i32 %x, 3
  %negated = sub nsw i32 0, %masked
  br label %loop
loop:
  %cycle = phi i32 [ %negated, %entry ], [ %next, %loop ]
  %next = add i32 %cycle, 4
  br i1 %continue, label %loop, label %exit
exit:
  ret i32 %masked
}

; `and` with 0 demands no bit of %u, but %u is still rebuilt, so that it drops
; its nsw flag. Kept as it is, it would receive %p narrowed to the 8 bits %low
; demands, and shift out bits that disagree with its sign bit.
define i8 @zero_demand_flags(i64 %x, i64 %y, ptr %out) {
  ; RAW-LABEL: @zero_demand_flags(
  ; RAW-NOT: nsw
  ; RAW: shl i64 %{{[^ ]+}}, 56
  ; RAW-NOT: nsw
  ; RAW: ret i8
  %p = add i64 %x, %y
  %low = trunc i64 %p to i8
  %u = shl nsw i64 %p, 56
  %z = and i64 %u, 0
  store i64 %z, ptr %out, align 8
  ret i8 %low
}

; Execution needs both the demanded bits and those shifted into them. The
; exact flag must not migrate to the narrowed computation.
define i8 @exact_shift(i64 %x) {
  ; CHECK-LABEL: @exact_shift(
  ; CHECK: lshr i16 {{.*}}, 8
  %r = lshr exact i64 %x, 8
  %low = trunc i64 %r to i8
  ret i8 %low
}

define i128 @wide_source(i8 %x, i8 %y) {
  ; CHECK-LABEL: @wide_source(
  ; CHECK: add i16
  ; CHECK: zext i16 {{.*}} to i128
  %a = zext i8 %x to i128
  %b = zext i8 %y to i128
  %r = add i128 %a, %b
  ret i128 %r
}

define <2 x i64> @vector_unchanged(<2 x i64> %x, <2 x i64> %y) {
  ; CHECK-LABEL: @vector_unchanged(
  ; CHECK: add <2 x i64>
  %r = add <2 x i64> %x, %y
  ret <2 x i64> %r
}

; A function with an `invoke` is left alone, including the computations the
; `invoke` is not involved in.
define i8 @with_invoke(i64 %x, i64 %y) personality ptr @personality {
  ; RAW-LABEL: @with_invoke(
  ; RAW: %sum = add i64 %x, %y
  ; RAW: %low = trunc i64 %sum to i8
entry:
  %sum = add i64 %x, %y
  %low = trunc i64 %sum to i8
  %unused = invoke i64 @may_throw() to label %normal unwind label %unwind
normal:
  ret i8 %low
unwind:
  %exception = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %exception
}

declare i64 @may_throw()
declare i32 @personality(...)

; The common select demand reaches its i1 condition without keeping data wide.
define i64 @select_condition(i8 %x, i8 %y) {
  ; CHECK-LABEL: @select_condition(
  ; CHECK: icmp ult i8
  ; CHECK: select i1 {{.*}}, i8 %x, i8 %y
  ; CHECK: zext i8 {{.*}} to i64
  %a = zext i8 %x to i64
  %b = zext i8 %y to i64
  %condition = icmp slt i64 %a, 200
  %selected = select i1 %condition, i64 %a, i64 %b
  ret i64 %selected
}

; Value demand is 8 but count demand is 16. Keep their transfers separate.
define i8 @shift_roles(i64 %x, i64 %y, i64 %amount) {
  ; CHECK-LABEL: @shift_roles(
  ; CHECK: add i8
  ; CHECK: shl i16
  ; CHECK: trunc i16 {{.*}} to i8
  %value = add i64 %x, %y
  %count = and i64 %amount, 15
  %shifted = shl i64 %value, %count
  %result = trunc i64 %shifted to i8
  ret i8 %result
}

; A range-aware backward transfer needs only 15 bits of the addition. LLVM
; 16's DemandedBits requests all 64 bits through this variable shift.
define i8 @bounded_shift_producer(i64 %x, i64 %y, i64 %count) {
  ; RAW-LABEL: @bounded_shift_producer(
  ; RAW: add i16
  ; RAW: lshr i16
  %value = add i64 %x, %y
  %amount = and i64 %count, 7
  %shifted = lshr i64 %value, %amount
  %low = trunc i64 %shifted to i8
  ret i8 %low
}
