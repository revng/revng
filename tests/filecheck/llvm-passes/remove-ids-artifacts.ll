;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %root/bin/revng opt %s -remove-ids-artifacts -S -o - | FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; Every function here has a single exit on the `ScopeGraph`, which is what
; `enforce-single-exit` guarantees where the pass runs and what the pass
; asserts on entry. The divergent exits are `goto` blocks carrying a
; `scope_closer`, which is the shape IDS leaves behind: a `goto` block exposes
; its `scope_closer` target as its only `ScopeGraph` successor, so those
; markers are retargeted and never dropped.
;
; Each chain link is closed into by a `scope_closer`, which is what makes it an
; IDS `Tail` rather than an IDS `Head`. The functions that must not be
; transformed carry that marker too, so that the only thing disqualifying them
; is the property each one is named for.

; The shape IDS leaves behind: two blocks switching on the same value, the
; second holding nothing but its switch, and a `goto` block closing its scope
; into it. The chain is the one switch it started as, so it collapses back into
; `head`, and the `scope_closer`, whose target is about to disappear, is aimed
; at the exit instead.

define void @two_switches(i32 %v) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %v, label %exit [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@two_switches, %tail0))
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @two_switches(i32 %v)
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %exit [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT: ]
; CHECK: a:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@two_switches, %exit))
; CHECK-NOT: tail0

; Three rounds of IDS stack three switches. Resolving a value walks the whole
; chain: 2 is handled by neither of the first two switches, so it falls through
; both defaults before `tail1` dispatches it. Both `scope_closer`s move to the
; exit, including the one that pointed at the middle of the chain.

define void @three_switches(i32 %v) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %v, label %tail1 [
    i32 1, label %b
  ]

tail1:
  switch i32 %v, label %exit [
    i32 2, label %c
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@three_switches, %tail0))
  br label %exit

b:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@three_switches, %tail1))
  br label %exit

c:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @three_switches(i32 %v)
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %exit [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT:   i32 2, label %c
; CHECK-NEXT: ]
; CHECK: a:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@three_switches, %exit))
; CHECK: b:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@three_switches, %exit))
; CHECK-NOT: tail0
; CHECK-NOT: tail1

; Where a retargeted `scope_closer` lands. Every other function here collapses
; straight into the exit, so the block a `scope_closer` has to be moved to and
; the exit of the `ScopeGraph` are the same block, and nothing distinguishes
; aiming at one from aiming at the other. Here they differ: the two successors
; of `tail0` reconverge on `join`, which postdominates `tail0` without being
; the exit, so the `scope_closer` in `a` has to end up on `join`.
;
; Aiming it at `exit` instead keeps the graph single exit and looks harmless,
; but it widens the scope the folded `switch` opens to the end of the function.
; `join` then postdominates nothing, sits inside that scope with two arms
; reaching it, and the clifter emits it once per arm. Measured on 2026-11-20:
; that aborts `emit-c` on five of the seven measurement binaries, in
; `Clifter.cpp`, on `not Iterator->second.InsertPoint.isSet()`.

define void @closer_lands_on_the_postdominator(i32 %v) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %v, label %c [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@closer_lands_on_the_postdominator, %tail0))
  br label %exit

b:
  br label %join

c:
  br label %join

join:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @closer_lands_on_the_postdominator(i32 %v)
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %c [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT: ]
; CHECK: a:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@closer_lands_on_the_postdominator, %join))
; CHECK-NOT: tail0

; A block with exactly the shape of a chain link that nothing closes into is an
; IDS `Head`, not a `Tail`. `createHead` builds it by cloning the same
; terminator, so it is a lone switch on the same condition reached from the
; conditional alone, but it exists to give a divergent scope entered through
; several successors one entry that dominates its postdominator. Erasing it
; would hand those successors back to the conditional, so it is left alone.

define void @ids_head_is_not_a_tail(i32 %v) {
head:
  switch i32 %v, label %exit [
    i32 0, label %divergent_head
  ]

divergent_head:
  switch i32 %v, label %a [
    i32 1, label %b
  ]

a:
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @ids_head_is_not_a_tail(i32 %v)
; CHECK: switch i32 %v, label %exit [
; CHECK: divergent_head:
; CHECK-NEXT: switch i32 %v, label %a [

; The chain does not have to be entered through a default. Here `head` hands
; value 0 over to `tail0` through a case, so 0 has to resolve to where `tail0`
; sends it, `d`, and not to `tail0` itself. The same value labels both switches,
; so it also checks that the merged switch mentions it exactly once, which LLVM
; requires.

define void @case_edge_into_chain(i32 %v) {
head:
  switch i32 %v, label %a [
    i32 0, label %tail0
    i32 1, label %b
  ]

tail0:
  switch i32 %v, label %a [
    i32 0, label %d
  ]

a:
  br label %exit

b:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@case_edge_into_chain, %tail0))
  br label %exit

d:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @case_edge_into_chain(i32 %v)
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %a [
; CHECK-NEXT:   i32 0, label %d
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT: ]
; CHECK-NOT: tail0

; A `scope_closer` aimed at the head is left where it is: the head survives the
; merge, so its target does not dangle and there is nothing to retarget.

define void @closer_on_head(i32 %v, i1 %c) {
entry:
  br i1 %c, label %head, label %closing

closing:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@closer_on_head, %head))
  br label %exit

head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %v, label %exit [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@closer_on_head, %tail0))
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @closer_on_head(i32 %v, i1 %c)
; CHECK: closing:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@closer_on_head, %head))
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %exit [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT: ]

; A `scope_closer` aimed at the chain link from outside the region its
; predecessor dominates. `outside` closes into `tail0`, but it is reached
; without going through `head`, so folding `tail0` into `head` would retarget a
; marker belonging to a path the fold has nothing to do with. Everything else
; about the shape qualifies, so the dominance of the sources is the only thing
; holding the fold back and `tail0` survives.

define void @closer_source_outside_the_predecessor(i32 %v, i1 %c) {
entry:
  br i1 %c, label %head, label %outside

outside:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@closer_source_outside_the_predecessor, %tail0))
  br label %exit

head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %v, label %exit [
    i32 1, label %b
  ]

a:
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @closer_source_outside_the_predecessor(i32 %v, i1 %c)
; CHECK: outside:
; CHECK-NEXT: call void @goto_block()
; CHECK-NEXT: call void @scope_closer(ptr blockaddress(@closer_source_outside_the_predecessor, %tail0))
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %tail0 [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT: ]
; CHECK: tail0:
; CHECK-NEXT: switch i32 %v, label %exit [
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT: ]

; Two switches on different values are not a chain, however they are laid out:
; merging them would have to pick one condition and would change what the
; function computes.

define void @different_condition(i32 %v, i32 %w) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  switch i32 %w, label %exit [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@different_condition, %tail0))
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @different_condition(i32 %v, i32 %w)
; CHECK: switch i32 %v, label %tail0 [
; CHECK: switch i32 %w, label %exit [

; A second block that carries an instruction besides its switch is not an IDS
; artifact, and erasing it would take that instruction with it.

define void @tail_not_empty(i32 %v) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

tail0:
  %sum = add i32 %v, 1
  switch i32 %v, label %exit [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@tail_not_empty, %tail0))
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @tail_not_empty(i32 %v)
; CHECK: switch i32 %v, label %tail0 [
; CHECK: tail0:
; CHECK-NEXT: %sum = add i32 %v, 1
; CHECK-NEXT: switch i32 %v, label %exit [

; A second block reached by something other than the first one cannot be
; erased: `other` would lose its only path forward.

define void @shared_tail(i32 %v, i1 %c) {
entry:
  br i1 %c, label %head, label %other

head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
  ]

other:
  br label %tail0

tail0:
  switch i32 %v, label %exit [
    i32 1, label %b
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@shared_tail, %tail0))
  br label %exit

b:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @shared_tail(i32 %v, i1 %c)
; CHECK: switch i32 %v, label %tail0 [
; CHECK: tail0:
; CHECK-NEXT: switch i32 %v, label %exit [

; The shape real input actually has. IDS redirects the edges it moves, it does
; not drop the labels that carried them, so every link keeps every case label
; and the ones a link has already dealt with point at the next link. A value
; therefore arrives at a link either through the default, which carries
; everything the predecessor does not label, or through a case that pins the
; value down, and either way the link is the one that dispatches it. Folding
; the two gives back the switch IDS started from.

define void @overlapping_labels(i32 %v) {
head:
  switch i32 %v, label %tail0 [
    i32 0, label %a
    i32 1, label %tail0
    i32 2, label %tail0
  ]

tail0:
  switch i32 %v, label %exit [
    i32 0, label %exit
    i32 1, label %b
    i32 2, label %c
  ]

a:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@overlapping_labels, %tail0))
  br label %exit

b:
  br label %exit

c:
  br label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @overlapping_labels(i32 %v)
; CHECK: head:
; CHECK-NEXT: switch i32 %v, label %exit [
; CHECK-NEXT:   i32 0, label %a
; CHECK-NEXT:   i32 1, label %b
; CHECK-NEXT:   i32 2, label %c
; CHECK-NEXT: ]
; CHECK-NOT: tail0
