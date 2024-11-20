;
; Copyright rev.ng Labs Srl. See LICENSE.md for details.
;

; RUN: %revngopt %s -generic-region-info -debug-log=generic-region-info -o /dev/null |& FileCheck %s

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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: while.cond
; CHECK-NEXT: while.cond
; CHECK-NEXT: while.body

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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: do.body
; CHECK-NEXT: do.body
; CHECK-NEXT: do.cond

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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: while.cond
; CHECK-NEXT: while.cond
; CHECK-NEXT: while.end
; CHECK-NEXT: while.cond1
; CHECK-NEXT: while.body3
; CHECK-NEXT: while.body
; CHECK: Region 1:
; CHECK-NEXT: Elected head: while.cond1
; CHECK-NEXT: while.cond1
; CHECK-NEXT: while.body3

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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: while.cond
; CHECK-NEXT: while.cond
; CHECK-NEXT: if.then
; CHECK-NEXT: label
; CHECK-NEXT: if.then4
; CHECK-NEXT: if.end
; CHECK-NEXT: while.body
; CHECK-NEXT: if.end5
; CHECK: Region 1:
; CHECK-NEXT: Elected head: label
; CHECK-NEXT: label
; CHECK-NEXT: if.then4
; CHECK-NEXT: if.end

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

; CHECK-LABEL: Generic Region Info Results:

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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: while.cond
; CHECK-NEXT: while.cond
; CHECK-NEXT: while.body
; CHECK: Region 1:
; CHECK-NEXT: Elected head: while.body
; CHECK-NEXT: while.body

; double edge test

define void @n() #0 {
block_a:
  br i1 undef, label %block_b, label %block_b

block_b:
  ret void
}

; CHECK-LABEL: Generic Region Info Results:

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

; CHECK-LABEL: Generic Region Info Results:

; multiple exit nodes test

define void @p() #0 {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  ret void
}

; CHECK-LABEL: Generic Region Info Results:


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

; CHECK-LABEL: Generic Region Info Results:


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

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: block_b
; CHECK-NEXT: block_b
; CHECK-NEXT: block_d
; CHECK-NEXT: block_c
