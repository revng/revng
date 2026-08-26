//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// Neither branch falls through. The then ends in a goto and the else in a
// break (neither continue nor return): break has higher precedence, so the
// else is hoisted and the then keeps its goto.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    %label = clift.make_label
    clift.while break %break cond {
      %c = clift.imm 1 : !int32_t
      clift.yield %c : !int32_t
    } body {
      // CHECK: clift.if {
      clift.if {
        clift.yield %arg0 : !int32_t
      // CHECK: } then {
      // CHECK-NEXT: clift.goto
      } then {
        clift.goto %label
      // CHECK-NOT: else
      } else {
        clift.break_to %break
      }
      // CHECK: clift.break_to
      clift.assign_label %label
    }
  }
}
