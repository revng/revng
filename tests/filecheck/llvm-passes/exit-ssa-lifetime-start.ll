;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -exit-ssa -S -o - | FileCheck %s

; `exit-ssa` marks where each variable it creates becomes live with a
; `llvm.lifetime.start`, which is what the clifter uses to decide where to
; declare it. The marker goes at the end of the nearest common dominator of all
; the incoming blocks of all the PHIs in the class, which is what puts it
; outside any loop the variable is carried across.

; A variable that is undefined when entering the loop and carried across
; iterations. Every access to it is inside the loop, so the dominator of the
; accesses alone would sit in the loop body and the variable would be declared
; there, picking up an indeterminate value on each iteration. The incoming
; blocks are `entry` and `loop`, whose dominator is `entry`, so the marker lands
; outside the loop.
;
; Note also that the undefined incoming produces no store at all: the marker is
; what carries the declaration point, so there is nothing to write.

define i32 @loop_carried(i32 %n) {
entry:
  br label %loop

loop:
  %acc = phi i32 [ undef, %entry ], [ %next, %loop ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %next = add i32 %acc, %i
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret i32 %next
}

; CHECK-LABEL: define i32 @loop_carried(i32 %n)
; CHECK: entry:
; CHECK-NEXT: %[[ACC:[0-9]+]] = alloca i32
; CHECK-NEXT: %[[I:[0-9]+]] = alloca i32
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %[[ACC]])
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %[[I]])
; CHECK-NEXT: store i32 0, ptr %[[I]]
; CHECK-NEXT: br label %loop
; CHECK: loop:
; The marker is not in the loop, and the undefined incoming stores nothing:
; there is no value to write, and the declaration point is already carried.
; CHECK-NOT: lifetime.start
; CHECK-NOT: store i32 undef

; The marker is placed at the dominator of the incoming blocks, not at a single
; incoming block, so for a diamond it ends up above the branch rather than in
; either arm.

define i32 @diamond(i1 %c, i32 %a, i32 %b) {
entry:
  br i1 %c, label %left, label %right

left:
  br label %join

right:
  br label %join

join:
  %v = phi i32 [ %a, %left ], [ %b, %right ]
  ret i32 %v
}

; CHECK-LABEL: define i32 @diamond(i1 %c, i32 %a, i32 %b)
; CHECK: entry:
; CHECK-NEXT: %[[V:[0-9]+]] = alloca i32
; CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 {{[^,]+}}, ptr %[[V]])
; CHECK-NEXT: br i1 %c, label %left, label %right
; CHECK: left:
; CHECK-NEXT: store i32 %a, ptr %[[V]]
; CHECK: right:
; CHECK-NEXT: store i32 %b, ptr %[[V]]
