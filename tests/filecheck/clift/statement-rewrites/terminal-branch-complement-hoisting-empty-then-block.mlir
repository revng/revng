//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --terminal-branch-complement-hoisting | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.if {
    clift.if {
      // CHECK: %0 = clift.test %arg0 : !int32_t
      // CHECK: %1 = clift.not %0
      // CHECK: clift.yield %1 : !clift.bool
      %0 = clift.test %arg0 : !int32_t
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
      ^bb0:
      // CHECK: clift.expr {
        // CHECK: %0 = clift.imm 10 : !int32_t
        // CHECK: clift.yield %0 : !int32_t
      // CHECK: }
    // CHECK-NOT: } else {
    } else {
      clift.expr {
        %0 = clift.imm 10 : !int32_t
        clift.yield %0 : !int32_t
      }
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
