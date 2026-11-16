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
      // CHECK: clift.for
      // CHECK: body {
      clift.for break %0 body {
        // CHECK-NOT: clift.break_to
        // CHECK: clift.return {
        // CHECK: }
        clift.break_to %0
      // CHECK: }
      }
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
