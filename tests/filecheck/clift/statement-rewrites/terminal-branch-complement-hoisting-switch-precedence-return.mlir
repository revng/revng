//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// No branch falls through. The default ends in a return and the case in a
// break: return has higher precedence, so the default is hoisted and the case
// keeps its break.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    clift.while break %break cond {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } body {
      // CHECK: clift.switch {
      clift.switch {
        clift.yield %arg0 : !int32_t
      // CHECK: } case 0 {
      // CHECK-NEXT: clift.break_to
      } case 0 {
        clift.break_to %break
      // CHECK-NOT: default
      } default {
        clift.return {}
      }
      // CHECK: clift.return
    }
  }
}
