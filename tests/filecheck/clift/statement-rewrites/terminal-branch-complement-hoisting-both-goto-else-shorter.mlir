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
    %0 = clift.make_label
    clift.assign_label %0
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.test %arg0 : !int32_t
      %1 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield %1 : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %1 = clift.imm 10 : !int32_t
        %1 = clift.imm 10 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %1 = clift.imm 20 : !int32_t
        %1 = clift.imm 20 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.goto %0
      clift.goto %0
    // CHECK-NOT: } else {
    } else {
      clift.expr {
        %1 = clift.imm 30 : !int32_t
        clift.yield %1 : !int32_t
      }
      clift.goto %0
    // CHECK: }
    }
    // CHECK: clift.expr {
      // CHECK: %1 = clift.imm 30 : !int32_t
      // CHECK: clift.yield %1 : !int32_t
    // CHECK: }
    // CHECK: clift.goto %0
  // CHECK: }
  }
// CHECK: }
}
