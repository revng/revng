//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --hoist-terminal-branches | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.assign_label %0
    clift.assign_label %0
    // COM: no break can leave the loop closing the else-branch, so neither
    // COM: branch falls through and the heavier one is hoisted
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.test %arg0 : !int32_t
      %1 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield %1 : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.goto %0
      clift.goto %0
    // CHECK: }
    // CHECK-NOT: else
    } else {
      clift.expr {
        %1 = clift.imm 30 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.while cond {
        %1 = clift.true
        clift.yield %1 : !clift.bool
      } body {
        clift.expr {
          %1 = clift.imm 40 : !int32_t
          clift.yield %1 : !int32_t
        }
      }
    }
    // CHECK: clift.expr {
      // CHECK: %1 = clift.imm 30 : !int32_t
      // CHECK: clift.yield %1 : !int32_t
    // CHECK: }
    // CHECK: clift.while cond {
      // CHECK: %1 = clift.true
      // CHECK: clift.yield %1 : !clift.bool
    // CHECK: } body {
      // CHECK: clift.expr {
        // CHECK: %1 = clift.imm 40 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
      // CHECK: }
    // CHECK: }
  // CHECK: }
  }
// CHECK: }
}
