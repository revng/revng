;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -generic-region-info -debug-log=generic-region-info -o /dev/null |& FileCheck %s

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

; head election with three candidates. `five_incoming` must win, since it has
; the highest number of incoming edges from outside the region. It is visited
; before `three_incoming` in RPOT, and the first candidate of the region is
; `one_incoming`, so a maximum that is not kept up to date would let the later
; `three_incoming` overwrite the correct head.

define void @s() #0 {
entry:
  switch i32 undef, label %o1 [ i32 1, label %o2
                                i32 2, label %o3
                                i32 3, label %o4
                                i32 4, label %o5
                                i32 5, label %o6
                                i32 6, label %o7
                                i32 7, label %o8
                                i32 8, label %o9 ]

o1:
  br label %three_incoming

o2:
  br label %three_incoming

o3:
  br label %three_incoming

o4:
  br label %five_incoming

o5:
  br label %five_incoming

o6:
  br label %five_incoming

o7:
  br label %five_incoming

o8:
  br label %five_incoming

o9:
  br label %one_incoming

no_incoming:
  br label %three_incoming

three_incoming:
  br label %five_incoming

five_incoming:
  br label %one_incoming

one_incoming:
  br i1 undef, label %no_incoming, label %exit

exit:
  ret void
}

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: five_incoming

; a head candidate whose successors are all late entries of a child region
; cannot be elected: after the child region is dagified those successors are
; only reachable through `goto`s, so such a head would not reach the rest of
; the region. Here `only_reaches_inner` must be discarded in favour of
; `outer_head`.

define void @t() #0 {
entry:
  br i1 undef, label %only_reaches_inner, label %other_entry

only_reaches_inner:
  br label %inner_late_entry

other_entry:
  br label %outer_head

inner_pred:
  br i1 undef, label %inner_head, label %inner_head

inner_late_entry:
  br label %inner_head

outer_head:
  br i1 undef, label %only_reaches_inner, label %inner_pred

inner_head:
  br i1 undef, label %outer_head, label %inner_late_entry
}

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: outer_head

; the check on whether a head candidate reaches the whole region must be
; transitive. Here `reaches_only_late_entry` is not a late entry of the child
; region itself, and neither is its only successor `late_entry_pred`, but
; `late_entry_pred` only leads to `inner_late_entry`, which is one. So
; `reaches_only_late_entry` would reach nothing but `late_entry_pred` once the
; child region is dagified, and `outer_head` must be elected instead.

define void @u() #0 {
entry:
  br i1 undef, label %reaches_only_late_entry, label %other_entry

reaches_only_late_entry:
  br label %late_entry_pred

late_entry_pred:
  br label %inner_late_entry

other_entry:
  br label %outer_head

inner_late_entry:
  br label %inner_head

outer_head:
  br i1 undef, label %reaches_only_late_entry, label %inner_head

inner_head:
  br i1 undef, label %outer_head, label %inner_late_entry
}

; CHECK-LABEL: Generic Region Info Results:
; CHECK: Region 0:
; CHECK-NEXT: Elected head: outer_head
