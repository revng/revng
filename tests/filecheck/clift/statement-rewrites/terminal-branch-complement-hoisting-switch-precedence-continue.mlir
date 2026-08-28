//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// No branch falls through, so any could be hoisted. The default ends in a
// continue and the case in a return: continue has higher precedence, so the
// default is hoisted and the case keeps its return.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %break = clift.make_label
    %continue = clift.make_label
    clift.while break %break continue %continue cond {
      %0 = clift.true
      clift.yield %0 : !clift.bool
    } body {
      // CHECK: clift.switch {
      clift.switch {
        clift.yield %arg0 : !int32_t
      // CHECK: } case 0 {
      // CHECK-NEXT: clift.return
      } case 0 {
        clift.return {}
      // CHECK-NOT: default
      } default {
        clift.continue_to %continue
      }
      // CHECK: clift.continue_to
    }
  }
}
