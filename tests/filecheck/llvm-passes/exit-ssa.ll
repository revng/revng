;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -exit-ssa -S -o - | FileCheck %s

;
; CHECK-LABEL: define i32 @double_swap(i32 %a, i32 %b)
;
define i32 @double_swap(i32 %a, i32 %b) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA0:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK: [[ALLOCA1:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK: store i32 %b, ptr [[ALLOCA0]]
; CHECK: store i32 %a, ptr [[ALLOCA1]]
; CHECK: br
  br label %while.cond

; CHECK: while.cond:
while.cond:
  ; CHECK: [[LOAD1:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA1]]
  ; CHECK: [[LOAD0:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA0]]
  ; CHECK: store i32 [[LOAD0]], ptr [[ALLOCA1]]
  ; CHECK: store i32 [[LOAD1]], ptr [[ALLOCA0]]
  %b.addr.0 = phi i32 [ %b, %entry ], [ %a.addr.0, %while.cond ]
  %a.addr.0 = phi i32 [ %a, %entry ], [ %b.addr.0, %while.cond ]
  br i1 undef, label %while.end, label %while.cond
; CHECK: br

; CHECK: while.end:
while.end:
  ; CHECK: sub i32 [[LOAD1]], [[LOAD0]]
  %sub = sub i32 %a.addr.0, %b.addr.0
  ret i32 %sub
}

;
; CHECK-LABEL: define i32 @lost_copy(i32 %a)
;
define i32 @lost_copy(i32 %a) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
; CHECK: store i32 %a, ptr [[ALLOCA]]
  br label %while.cond
; CHECK: br

; CHECK: while.cond:
while.cond:
  ; CHECK: [[LOAD:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA]]
  ; CHECK: [[ADD:%[a-zA-Z0-9_]+]] = add i32 [[LOAD]], 1
  ; CHECK: store i32 [[ADD]], ptr [[ALLOCA]]
  %b = phi i32 [ %a, %entry ], [ %c, %while.cond ]
  %c = add i32 %b, 1
  br i1 undef, label %while.end, label %while.cond
; CHECK: br

; CHECK: while.end:
while.end:
  ; CHECK: ret i32 [[LOAD]]
  ret i32 %b
}

;
; CHECK-LABEL: define i32 @chain(i32 %a, i32 %b, i32 %c)
;
; Two distinct PHINodes where the first (%p1) is an incoming of the second
; (%p2), and both also have non-PHI users (%u1 and %u2). They are collapsed
; onto the same alloca.
define i32 @chain(i32 %a, i32 %b, i32 %c) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
  br i1 undef, label %A, label %B

; CHECK: A:
A:
; CHECK: store i32 %a, ptr [[ALLOCA]]
  br label %M1

; CHECK: B:
B:
; CHECK: store i32 %b, ptr [[ALLOCA]]
  br label %M1

; CHECK: M1:
M1:
; CHECK: [[LOAD1:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA]]
; CHECK: add i32 [[LOAD1]], 1
  %p1 = phi i32 [ %a, %A ], [ %b, %B ]
  %u1 = add i32 %p1, 1
  br i1 undef, label %C, label %M2

; CHECK: C:
C:
; CHECK: store i32 %c, ptr [[ALLOCA]]
  br label %M2

; CHECK: M2:
M2:
; CHECK: [[LOAD2:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA]]
; CHECK: add i32 [[LOAD2]], 2
  %p2 = phi i32 [ %p1, %M1 ], [ %c, %C ]
  %u2 = add i32 %p2, 2
  %r = add i32 %u1, %u2
  ret i32 %r
}

;
; CHECK-LABEL: define i32 @cone(i32 %a, i32 %b, i32 %c, i32 %d)
;
; A cone (DAG) of PHINodes coming from a nested if-else: %p1 and %p2 are both
; incomings of %pt. The whole cone is collapsed onto a single alloca, with no
; edge splitting (each incoming block contributes a single value).
define i32 @cone(i32 %a, i32 %b, i32 %c, i32 %d) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
  br i1 undef, label %if1, label %else1

; CHECK: if1:
if1:
; CHECK: store i32 %a, ptr [[ALLOCA]]
  br i1 undef, label %m1, label %f1

; CHECK: f1:
f1:
; CHECK: store i32 %b, ptr [[ALLOCA]]
  br label %m1

m1:
  %p1 = phi i32 [ %a, %if1 ], [ %b, %f1 ]
  br label %exit

; CHECK: else1:
else1:
; CHECK: store i32 %c, ptr [[ALLOCA]]
  br i1 undef, label %m2, label %f2

; CHECK: f2:
f2:
; CHECK: store i32 %d, ptr [[ALLOCA]]
  br label %m2

m2:
  %p2 = phi i32 [ %c, %else1 ], [ %d, %f2 ]
  br label %exit

; CHECK: exit:
exit:
; CHECK: load i32, ptr [[ALLOCA]]
  %pt = phi i32 [ %p1, %m1 ], [ %p2, %m2 ]
  %u = add i32 %pt, 0
  ret i32 %u
}

;
; CHECK-LABEL: define i32 @edgesplit(i32 %a, i32 %b, i32 %c, i1 %t)
;
; %p1 and %p2 belong to the same equivalence class, but from the same incoming
; block (%B1) they receive different values (%a for %p1, %b for %p2). Only one
; store can live in %B1, so the edges towards the other PHIs are split into new
; blocks where the remaining stores are emitted.
define i32 @edgesplit(i32 %a, i32 %b, i32 %c, i1 %t) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i32
  br i1 undef, label %B1, label %B2

; CHECK: B1:
B1:
; CHECK: store i32 %a, ptr [[ALLOCA]]
; CHECK: br i1 %t, label %J1, label %B1-to-J2
  br i1 %t, label %J1, label %J2

; CHECK: B2:
B2:
; CHECK: store i32 %b, ptr [[ALLOCA]]
; CHECK: br i1 %t, label %J1, label %B2-to-J2
  br i1 %t, label %J1, label %J2

J1:
  %p1 = phi i32 [ %a, %B1 ], [ %b, %B2 ]
  br label %exit

J2:
  %p2 = phi i32 [ %b, %B1 ], [ %c, %B2 ]
  br label %exit

; CHECK: exit:
exit:
; CHECK: load i32, ptr [[ALLOCA]]
  %pt = phi i32 [ %p1, %J1 ], [ %p2, %J2 ]
  %u = add i32 %pt, 0
  ret i32 %u
; The remaining stores live in the freshly split edge blocks.
; CHECK: B1-to-J2:
; CHECK: store i32 %b, ptr [[ALLOCA]]
; CHECK: br label %J2
; CHECK: B2-to-J2:
; CHECK: store i32 %c, ptr [[ALLOCA]]
; CHECK: br label %J2
}

;
; CHECK-LABEL: define i32 @conflicting(i32 %a1, i32 %a2, i32 %x, i32 %y)
;
; %a is an incoming of both %b and %c, but from different predecessors (%P1 for
; %b, %P2 for %c). %b and %c hold conflicting values, so %a, %b and %c cannot be
; merged into a single class: two allocas are needed.
define i32 @conflicting(i32 %a1, i32 %a2, i32 %x, i32 %y) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA0:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK: [[ALLOCA1:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
  br i1 undef, label %e1, label %e2

; CHECK: e1:
e1:
  ; CHECK: store i32 %a1, ptr [[ALLOCA1]]
  br label %B1

; CHECK: e2:
e2:
  ; CHECK: store i32 %a2, ptr [[ALLOCA1]]
  br label %B1

; CHECK: B1:
B1:
  ; CHECK: [[PHIA:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA1]]
  %a = phi i32 [ %a1, %e1 ], [ %a2, %e2 ]
  br i1 undef, label %P1, label %P2

; CHECK: P1:
P1:
  ; CHECK: store i32 [[PHIA]], ptr [[ALLOCA0]]
  ; CHECK: store i32 %y, ptr [[ALLOCA1]]
  br label %B2

; CHECK: P2:
P2:
  ; CHECK: store i32 %x, ptr [[ALLOCA0]]
  br label %B2

; CHECK: B2:
B2:
; The two PHINodes live in two distinct allocas.
; CHECK-DAG: load i32, ptr [[ALLOCA0]]
; CHECK-DAG: load i32, ptr [[ALLOCA1]]
  %b = phi i32 [ %a, %P1 ], [ %x, %P2 ]
  %c = phi i32 [ %y, %P1 ], [ %a, %P2 ]
  %r = add i32 %b, %c
  ret i32 %r
}

;
; CHECK-LABEL: define i32 @conflicting_2(i32 %a1, i32 %a2, i32 %x, i32 %x2)
;
; Like conflicting, but %a is an incoming of both %b and %c from the same predecessor
; (%P1). We still need two allocas, because %b and %c receive conflicting values
; (%x and %x2) from the other predecessor (%P2).
define i32 @conflicting_2(i32 %a1, i32 %a2, i32 %x, i32 %x2) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA0:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK: [[ALLOCA1:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
  br i1 undef, label %e1, label %e2

; CHECK: e1:
e1:
  ; CHECK: store i32 %a1, ptr [[ALLOCA1]]
  br label %B1

; CHECK: e2:
e2:
  ; CHECK: store i32 %a2, ptr [[ALLOCA1]]
  br label %B1

B1:
  ; CHECK: [[PHIA:%[a-zA-Z0-9_]+]] = load i32, ptr [[ALLOCA1]]
  %a = phi i32 [ %a1, %e1 ], [ %a2, %e2 ]
  br i1 undef, label %P1, label %P2

; CHECK: P1:
P1:
  ; CHECK: store i32 [[PHIA]], ptr [[ALLOCA0]]
  br label %B2

; CHECK: P2:
P2:
  ; CHECK: store i32 %x, ptr [[ALLOCA0]]
  ; CHECK: store i32 %x2, ptr [[ALLOCA1]]
  br label %B2

; CHECK: B2:
B2:
; The two PHINodes live in two distinct allocas.
; CHECK-DAG: load i32, ptr [[ALLOCA0]]
; CHECK-DAG: load i32, ptr [[ALLOCA1]]
  %b = phi i32 [ %a, %P1 ], [ %x, %P2 ]
  %c = phi i32 [ %a, %P1 ], [ %x2, %P2 ]
  %r = add i32 %b, %c
  ret i32 %r
}

;
; CHECK-LABEL: define i32 @non_conflicting(i32 %a1, i32 %a2, i32 %x)
;
; Like conflicting_2, but %b and %c receive the same values from every predecessor, so
; %a, %b and %c can all be grouped together and a single alloca is enough.
define i32 @non_conflicting(i32 %a1, i32 %a2, i32 %x) {
; CHECK: entry:
entry:
; CHECK: [[ALLOCA:%[a-zA-Z0-9_]+]] = alloca i32
; CHECK-NOT: alloca
  br i1 undef, label %e1, label %e2

; CHECK: e1:
e1:
  ; CHECK: store i32 %a1, ptr [[ALLOCA]]
  br label %B1

; CHECK: e2:
e2:
  ; CHECK: store i32 %a2, ptr [[ALLOCA]]
  br label %B1

B1:
  %a = phi i32 [ %a1, %e1 ], [ %a2, %e2 ]
  br i1 undef, label %P1, label %P2

; CHECK: P1:
P1:
  ; CHECK-NOT: store
  br label %B2

; CHECK: P2:
P2:
  ; CHECK: store i32 %x, ptr [[ALLOCA]]
  br label %B2

; CHECK: B2:
B2:
; Both PHINodes share the same, single alloca.
; CHECK: load i32, ptr [[ALLOCA]]
; CHECK: load i32, ptr [[ALLOCA]]
  %b = phi i32 [ %a, %P1 ], [ %x, %P2 ]
  %c = phi i32 [ %a, %P1 ], [ %x, %P2 ]
  %r = add i32 %b, %c
  ret i32 %r
}
