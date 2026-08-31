//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-statements | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!f = !clift.func<"/model-type/1001" : !void()>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.assign_label %0
    clift.assign_label %0
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.true
      %1 = clift.true
      // CHECK: clift.yield %1 : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %1 = clift.imm 1 : !int32_t
        %1 = clift.imm 1 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.goto %0
    // CHECK: } else {
    } else {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %1 = clift.imm 2 : !int32_t
        %1 = clift.imm 2 : !int32_t
        // CHECK: clift.yield %1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.goto %0
    // CHECK: }
    }
    // CHECK-NOT: clift.goto %0
    clift.goto %0
  // CHECK: }
  }
// CHECK: }
}
