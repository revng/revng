//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// Two branches (the two cases) fall through. Hoisting either would still leave
// the other reaching the hoisted code, so nothing is hoisted.

module attributes {clift.module} {
  // CHECK-LABEL: clift.func @f
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
      // CHECK: clift.imm 30 : !int32_t
      clift.expr {
        %0 = clift.imm 30 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: } case 1 {
    } case 1 {
      // CHECK: clift.imm 40 : !int32_t
      clift.expr {
        %0 = clift.imm 40 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: } default {
    } default {
      // CHECK: clift.return
      clift.return {}
    }
  }
}
