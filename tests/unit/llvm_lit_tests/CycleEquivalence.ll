;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -cycle-equivalence -debug-log=cycle-equivalence -o /dev/null |& FileCheck %s

; while test

define void @f(i32 noundef %a) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %cmp = icmp slt i32 %a, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: while.body,0 -> while.cond
; CHECK-NEXT: while.cond,0 -> while.body
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: while.cond,1 -> while.end
; CHECK-NEXT: entry,0 -> while.cond
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (while.body <-> while.cond,3), 1
; CHECK-NEXT: 1 => (while.end <-> entry,4), 1

; do-while test

define void @g(i32 noundef %a) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.cond, %entry
  br label %do.cond

do.cond:                                          ; preds = %do.body
  %cmp = icmp slt i32 %a, 10
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %do.cond
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: do.cond,1 -> do.end
; CHECK-NEXT: entry,0 -> do.body
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: do.cond,0 -> do.body
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: do.body,0 -> do.cond
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (do.end <-> entry,4), 1
; CHECK-NEXT: 1 => (do.cond <-> do.body,2), 1
; CHECK-NEXT: 2 => (do.cond <-> do.body,2), 2

; nested whiles test

define void @h(i32 noundef %a, i32 noundef %b) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.end, %entry
  %cmp = icmp slt i32 %a, 10
  br i1 %cmp, label %while.body, label %while.end6

while.body:                                       ; preds = %while.cond
  br label %while.cond1

while.cond1:                                      ; preds = %while.body3, %while.body
  %cmp2 = icmp slt i32 %b, 20
  br i1 %cmp2, label %while.body3, label %while.end

while.body3:                                      ; preds = %while.cond1
  br label %while.cond1

while.end:                                        ; preds = %while.cond1
  br label %while.cond

while.end6:                                       ; preds = %while.cond
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: while.body3,0 -> while.cond1
; CHECK-NEXT: while.cond1,0 -> while.body3
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: while.end,0 -> while.cond
; CHECK-NEXT: while.cond1,1 -> while.end
; CHECK-NEXT: while.body,0 -> while.cond1
; CHECK-NEXT: while.cond,0 -> while.body
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: while.cond,1 -> while.end6
; CHECK-NEXT: entry,0 -> while.cond
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (while.body3 <-> while.cond1,6), 1
; CHECK-NEXT: 1 => (while.end <-> while.cond,7), 1
; CHECK-NEXT: 2 => (while.end6 <-> entry,8), 1

; overlapping cycles test

define void @i(i32 noundef %a) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %if.end5, %if.then, %entry
  %cmp = icmp slt i32 %a, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  br label %label

label:                                            ; preds = %if.then4, %while.body
  %cmp1 = icmp eq i32 %a, 5
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %label
  br label %while.cond

if.end:                                           ; preds = %label
  %cmp3 = icmp eq i32 %a, 6
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.end
  br label %label

if.end5:                                          ; preds = %if.end
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: if.then,0 -> while.cond
; CHECK-NEXT: label,0 -> if.then
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: if.then4,0 -> label
; CHECK-NEXT: if.end,0 -> if.then4
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: if.end5,0 -> while.cond
; CHECK-NEXT: if.end,1 -> if.end5
; CHECK-NEXT: Bracket Equivalence Class ID: 3
; CHECK-NEXT: label,1 -> if.end
; CHECK-NEXT: Bracket Equivalence Class ID: 4
; CHECK-NEXT: while.body,0 -> label
; CHECK-NEXT: while.cond,0 -> while.body
; CHECK-NEXT: Bracket Equivalence Class ID: 5
; CHECK-NEXT: while.cond,1 -> while.end
; CHECK-NEXT: entry,0 -> while.cond
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (if.then <-> while.cond,6), 1
; CHECK-NEXT: 1 => (if.then4 <-> label,9), 1
; CHECK-NEXT: 2 => (if.end5 <-> while.cond,10), 1
; CHECK-NEXT: 3 => (if.end <-> label,12), 3
; CHECK-NEXT: 4 => (label <-> while.cond,13), 3
; CHECK-NEXT: 5 => (while.end <-> entry,11), 1

; diamond test

define void @l(i32 noundef %a) #0 {
entry:
  %cmp = icmp sgt i32 %a, 10
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  br label %if.end

if.else:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: entry,1 -> if.else
; CHECK-NEXT: if.else,0 -> if.end
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: if.then,0 -> if.end
; CHECK-NEXT: entry,0 -> if.then
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (if.else <-> entry,1), 1
; CHECK-NEXT: 1 => (if.end <-> entry,4), 2

; while self-loop test

define void @m(i32 noundef %a) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %cmp = icmp slt i32 %a, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %cmp2 = icmp slt i32 %a, 20
  br i1 %cmp2, label %while.cond, label %while.body

while.end:                                        ; preds = %while.cond
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: while.body,0 -> while.cond
; CHECK-NEXT: while.cond,0 -> while.body
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: while.body,1 -> while.body
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: while.cond,1 -> while.end
; CHECK-NEXT: entry,0 -> while.cond
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (while.body <-> while.cond,3), 1
; CHECK-NEXT: 1 => (while.body <-> while.body,4), 1
; CHECK-NEXT: 2 => (while.end <-> entry,5), 1

; double edge test

define void @n() #0 {
block_a:
  br i1 undef, label %block_b, label %block_b

block_b:
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: block_a,1 -> block_b
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: block_a,0 -> block_b
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (block_b <-> block_a,1), 1
; CHECK-NEXT: 1 => (block_b <-> block_a,1), 2

; double edge switch test

define void @o() #0 {
block_a:
  switch i32 undef, label %block_c [ i32 0, label %block_b
                                     i32 1, label %block_b ]

block_b:
  br label %block_c

block_c:
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: block_a,1 -> block_b
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: block_a,2 -> block_b
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: block_b,0 -> block_c
; CHECK-NEXT: Bracket Equivalence Class ID: 3
; CHECK-NEXT: block_a,0 -> block_c
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (block_b <-> block_a,1), 1
; CHECK-NEXT: 1 => (block_b <-> block_a,2), 1
; CHECK-NEXT: 2 => (block_b <-> block_a,2), 2
; CHECK-NEXT: 3 => (block_c <-> block_a,4), 3

; multiple exit nodes test

define void @p() #0 {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: block_a,1 -> block_c
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: block_a,0 -> block_b
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (block_c <-> block_a,1), 1
; CHECK-NEXT: 1 => (sink <-> block_a,4), 2

; multiple exit nodes switch test

define void @q() #0 {
block_a:
  switch i32 undef, label %block_d [ i32 0, label %block_b
                                     i32 1, label %block_c ]

block_b:
  ret void

block_c:
  ret void

block_d:
  ret void
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: block_a,1 -> block_b
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: block_a,2 -> block_c
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: block_a,0 -> block_d
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (block_b <-> block_a,1), 1
; CHECK-NEXT: 1 => (block_c <-> block_a,2), 1
; CHECK-NEXT: 2 => (sink <-> block_a,6), 3

; no exit nodes test

define dso_local void @r() #0 {
block_a:
  br label %block_b

block_b:
  br label %block_c

block_c:
  br label %block_d

block_d:
  br label %block_b
}

; CHECK-LABEL: Ordered Bracket Analysis Results:
; CHECK-NEXT: Bracket Equivalence Class ID: 0
; CHECK-NEXT: block_d,0 -> block_b
; CHECK-NEXT: Bracket Equivalence Class ID: 1
; CHECK-NEXT: block_c,0 -> block_d
; CHECK-NEXT: block_b,0 -> block_c
; CHECK-NEXT: Bracket Equivalence Class ID: 2
; CHECK-NEXT: block_a,0 -> block_b
; CHECK-LABEL: Class Bracket Correspondence:
; CHECK-NEXT: 0 => (block_d <-> block_b,3), 1
; CHECK-NEXT: 1 => (block_d <-> block_a,4), 2
; CHECK-NEXT: 2 => (block_d <-> block_a,4), 1
