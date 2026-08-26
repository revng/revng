//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// The switch has no default, so an unmatched value falls through it: this is an
// implicit fall-through branch with no body, so nothing can be hoisted even
// though every case is non-fallthrough.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
      // CHECK: clift.return
      clift.return {}
    // CHECK: } case 1 {
    } case 1 {
      // CHECK: clift.return
      clift.return {}
    }
    // CHECK-NOT: clift.expr
  }
}
