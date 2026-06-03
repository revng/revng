//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --label-merging | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: [[L:%[0-9]+]] = clift.make_label
    %L_0 = clift.make_label
    // CHECK-NOT: clift.make_label
    %L_1 = clift.make_label

    // CHECK: clift.if {
    clift.if {
      // CHECK: clift.yield %arg0 : !int32_t
      clift.yield %arg0 : !int32_t
    // CHECK: } then {
    } then {
      // CHECK: clift.goto [[L]]
      clift.goto %L_1
    // CHECK: }
    }

    // CHECK: clift.for
    // CHECK-SAME: break [[L]]
    // CHECK-SAME: body {
    clift.for break %L_0 body {
      // CHECK: clift.if {
      clift.if {
        // CHECK: clift.yield %arg0 : !int32_t
        clift.yield %arg0 : !int32_t
      // CHECK: } then {
      } then {
        // CHECK: clift.break_to [[L]]
        clift.break_to %L_0
      // CHECK: }
      }
    // CHECK: }
    }
    // CHECK-NOT: clift.assign_label
    clift.assign_label %L_1
  }
// CHECK: }
}
