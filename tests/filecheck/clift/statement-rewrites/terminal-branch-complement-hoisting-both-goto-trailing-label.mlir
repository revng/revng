//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// As in terminal-branch-complement-hoisting-both-goto-trailing-statement.mlir
// the else is hoisted, but here a label assignment follows the if. Being a jump
// target it keeps the statements past it reachable, so it ends the erasure and
// both it and the return survive.

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    %0 = clift.make_label
    %1 = clift.make_label
    clift.assign_label %0
    // CHECK: clift.if {
    clift.if {
      // CHECK: %2 = clift.test %arg0 : !int32_t
      %2 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield %2 : !clift.bool
      clift.yield %2 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK-NEXT: clift.goto %1
      clift.goto %1
    // CHECK-NEXT: }
    } else {
      clift.goto %0
    }
    // CHECK-NEXT: clift.goto %0
    // CHECK-NEXT: clift.assign_label %1
    // CHECK-NEXT: clift.return
    clift.assign_label %1
    clift.return {}
  }
}
