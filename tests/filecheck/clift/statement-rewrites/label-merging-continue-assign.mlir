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
      // CHECK: [[COND:%[0-9]+]] = clift.test %arg0 : !int32_t
      %0 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield [[COND]] : !clift.bool
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.goto [[L]]
      clift.goto %L_1
    // CHECK: }
    }

    // CHECK: clift.for
    // CHECK-SAME: continue [[L]]
    // CHECK-SAME: body {
    clift.for continue %L_0 body {
      // CHECK: clift.if {
      clift.if {
        // CHECK: [[COND:%[0-9]+]] = clift.test %arg0 : !int32_t
        %0 = clift.test %arg0 : !int32_t
        // CHECK: clift.yield [[COND]] : !clift.bool
        clift.yield %0 : !clift.bool
      // CHECK: } then {
      } then {
        // CHECK: clift.continue_to [[L]]
        clift.continue_to %L_0
      // CHECK: }
      }
      // CHECK-NOT: clift.assign_label
      clift.assign_label %L_1
    // CHECK: }
    }
  }
// CHECK: }
}
