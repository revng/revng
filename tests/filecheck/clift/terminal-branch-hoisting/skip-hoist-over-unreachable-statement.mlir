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
    // Neither branch of the if-statement falls through, which makes the return
    // statement following it unreachable. Hoisting the else-branch over it
    // would leave a return statement followed by another statement.
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
    // CHECK: } else {
    } else {
      // CHECK: clift.expr {
        // CHECK: %1 = clift.imm 30 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
      // CHECK: }
      clift.expr {
        %1 = clift.imm 30 : !int32_t
        clift.yield %1 : !int32_t
      }
      // CHECK: clift.return {
      // CHECK: }
      clift.return {}
    // CHECK: }
    }
    // CHECK: clift.return {
    // CHECK: }
    clift.return {}
  // CHECK: }
  }
// CHECK: }
}
