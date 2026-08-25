//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// No branch falls through, so the switch itself is non-fallthrough. Any branch
// may be hoisted; the heuristic picks one, moving its body after the switch and
// leaving that branch empty.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    // CHECK-NEXT: } default {
    } case 0 {
      clift.return {}
    } default {
      // CHECK: clift.return
      clift.return {}
    // CHECK: }
    }
    // A return is hoisted after the switch.
    // CHECK: clift.return
  }
}
