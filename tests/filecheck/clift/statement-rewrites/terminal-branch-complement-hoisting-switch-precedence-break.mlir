//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// No branch falls through. The default ends in a break and the case in a goto
// (neither continue, return, nor break): break has higher precedence, so the
// default is hoisted and the case keeps its goto.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    %label = clift.make_label
    clift.while break %break cond {
      %c = clift.imm 1 : !int32_t
      clift.yield %c : !int32_t
    } body {
      // CHECK: clift.switch {
      clift.switch {
        clift.yield %arg0 : !int32_t
      // CHECK: } case 0 {
      // CHECK-NEXT: clift.goto
      } case 0 {
        clift.goto %label
      // CHECK-NOT: default
      } default {
        clift.break_to %break
      }
      // CHECK: clift.break_to
      clift.assign_label %label
    }
  }
}
