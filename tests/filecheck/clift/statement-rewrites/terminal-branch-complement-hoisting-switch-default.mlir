//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// Every case is non-fallthrough and the default falls through, so the default
// body is hoisted after the switch and the (now empty) default is dropped.

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
    // CHECK-NOT: default
    } default {
      clift.expr {
        %0 = clift.imm 10 : !int32_t
        clift.yield %0 : !int32_t
      }
    }
    // CHECK: clift.expr {
    // CHECK: clift.imm 10 : !int32_t
  }
}
