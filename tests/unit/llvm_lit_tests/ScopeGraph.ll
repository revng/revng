;
; This file is distributed under the MIT License. See LICENSE.md for details.
;

; RUN: %revngopt -load tests/unit/libtest_scopegraph.so %s -scope-graph-logger -o /dev/null |& FileCheck %s

; function tags metadata needed for all the tests
declare !revng.tags !0 void @scope_closer(ptr)
declare !revng.tags !1 void @goto_block()
!0 = !{!"marker", !"scope-closer"}
!1 = !{!"marker", !"goto-block"}

; no dashed edge test

define void @f() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  ret void

block_c:
  br i1 undef, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: f
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_c
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: block_b
; CHECK-NEXT: block_c
; CHECK-NEXT: block_e

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:  [1] %block_a
; CHECK-NEXT:    [2] %block_b
; CHECK-NEXT:    [2] %block_c
; CHECK-NEXT:      [3] %block_e
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:  [1]  <<exit node>>
; CHECK-NEXT:    [2] %block_b
; CHECK-NEXT:    [2] %block_a
; CHECK-NEXT:    [2] %block_c
; CHECK-NEXT:    [2] %block_e
; CHECK-NEXT: Roots: %block_b %block_e


; scope edge test, scope closer b->b is seen in the scopegraph

define void @g() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  call void @scope_closer(ptr blockaddress(@g, %block_b))
  ret void

block_c:
  br i1 undef, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: g
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_c
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: block_b
; CHECK-NEXT: block_c
; CHECK-NEXT: block_e

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:   [1] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:       [3] %block_e
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:     [2] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT: Roots: %block_e %block_b

; goto edge test, b->c is not seen in the scopegraph

define void @h() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  call void @goto_block()
  br label %block_c

block_c:
  br i1 undef, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: h
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_c
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: block_b
; CHECK-NEXT: block_c
; CHECK-NEXT: block_e

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:   [1] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:       [3] %block_e
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:     [2] %block_a
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT: Roots: %block_b %block_e

; goto edge test, edge b->c is not seen in the scopegraph, but scope closer b->b
; is correctly seen

define void @i() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@i, %block_b))
  br label %block_c

block_c:
  br i1 undef, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: i
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_c
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: block_b
; CHECK-NEXT: block_c
; CHECK-NEXT: block_e

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:   [1] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:       [3] %block_e
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT:     [2] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT: Roots: %block_e %block_b

; depth first test with different order on scopegraph wrt. cfg. The plain DFS on
; the cfg, would lead to a,b,c,e, while on the scopegraph we obtain a,b,e,c.

define void @l() {
block_a:
  br i1 undef, label %block_b, label %block_c

block_b:
  call void @scope_closer(ptr blockaddress(@l, %block_e))
  ret void

block_c:
  br i1 undef, label %block_b, label %block_e

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: l
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_c
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:   block_b
; CHECK-NEXT:   block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: block_b
; CHECK-NEXT: block_e
; CHECK-NEXT: block_c

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:   [1] %block_a
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT:     [2] %block_c
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT:       [3] %block_b
; CHECK-NEXT:       [3] %block_a
; CHECK-NEXT:       [3] %block_c
; CHECK-NEXT: Roots: %block_e

; test containing a loop with `scope_closer` edges reflecting the application of
; dagify and materialize-loop-scope

define void @m(i1 noundef %a, i1 noundef %b) {

block_a:
  br i1 %a, label %goto_c, label %block_b

goto_c:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@m, %block_b))
  br label %block_c

block_b:
  call void @scope_closer(ptr blockaddress(@m, %block_e))
  br label %block_c

block_c:
  br label %block_d

block_d:
  br i1 %b, label %goto_b, label %block_e

goto_b:
  call void @goto_block()
  call void @scope_closer(ptr blockaddress(@m, %block_e))
  br label %block_b

block_e:
  ret void
}

; CHECK-LABEL: ScopeGraph of function: m
; CHECK-NEXT: Block block_a successors:
; CHECK-NEXT:  goto_c
; CHECK-NEXT:  block_b
; CHECK-NEXT: Block goto_c successors:
; CHECK-NEXT:  block_b
; CHECK-NEXT: Block block_b successors:
; CHECK-NEXT:  block_c
; CHECK-NEXT:  block_e
; CHECK-NEXT: Block block_c successors:
; CHECK-NEXT:  block_d
; CHECK-NEXT: Block block_d successors:
; CHECK-NEXT:  goto_b
; CHECK-NEXT:  block_e
; CHECK-NEXT: Block goto_b successors:
; CHECK-NEXT:  block_e
; CHECK-NEXT: Block block_e successors:
; CHECK-NEXT: Depth first order:
; CHECK-NEXT: block_a
; CHECK-NEXT: goto_c
; CHECK-NEXT: block_b
; CHECK-NEXT: block_c
; CHECK-NEXT: block_d
; CHECK-NEXT: goto_b
; CHECK-NEXT: block_e

; CHECK-LABEL: Inorder Dominator Tree
; CHECK-NEXT:   [1] %block_a
; CHECK-NEXT:     [2] %goto_c
; CHECK-NEXT:     [2] %block_b
; CHECK-NEXT:       [3] %block_c
; CHECK-NEXT:         [4] %block_d
; CHECK-NEXT:           [5] %goto_b
; CHECK-NEXT:       [3] %block_e
; CHECK-NEXT: Roots: %block_a

; CHECK-LABEL: Inorder PostDominator Tree
; CHECK-NEXT:   [1]  <<exit node>>
; CHECK-NEXT:     [2] %block_e
; CHECK-NEXT:       [3] %block_b
; CHECK-NEXT:         [4] %goto_c
; CHECK-NEXT:         [4] %block_a
; CHECK-NEXT:       [3] %goto_b
; CHECK-NEXT:       [3] %block_d
; CHECK-NEXT:         [4] %block_c
; CHECK-NEXT: Roots: %block_e
