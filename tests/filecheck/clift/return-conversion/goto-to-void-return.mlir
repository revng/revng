//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --convert-jump-to-void-return | FileCheck %s

!void = !clift.void
!f = !clift.func<"/model-type/1001" : !void()>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>() -> !void {
    %0 = clift.make_label
    // CHECK: clift.if {
    clift.if {
      // CHECK: [[CONDITION:%[0-9]+]] = clift.true
      %1 = clift.true
      // CHECK: clift.yield [[CONDITION]] : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK-NOT: clift.goto
      // CHECK: clift.return {
      // CHECK: }
      clift.goto %0
    // CHECK: }
    }
    clift.assign_label %0
  // CHECK: }
  }
// CHECK: }
}
