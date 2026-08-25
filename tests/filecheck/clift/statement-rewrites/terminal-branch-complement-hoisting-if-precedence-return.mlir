//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// Neither branch falls through. The then ends in a break and the else in a
// return: return has higher precedence, so the else is hoisted and the then
// keeps its break.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    clift.while break %break cond {
      %c = clift.imm 1 : !int32_t
      clift.yield %c : !int32_t
    } body {
      // CHECK: clift.if {
      clift.if {
        clift.yield %arg0 : !int32_t
      // CHECK: } then {
      // CHECK-NEXT: clift.break_to
      } then {
        clift.break_to %break
      // CHECK-NOT: else
      } else {
        clift.return {}
      }
      // CHECK: clift.return
    }
  }
}
