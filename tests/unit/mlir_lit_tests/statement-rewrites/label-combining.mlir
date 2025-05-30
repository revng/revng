//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --beautify-statements | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: [[L:%[0-9]+]] = clift.make_label
    %L_0 = clift.make_label
    %L_1 = clift.make_label

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.goto [[L]]
      clift.goto %L_0
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.goto [[L]]
      clift.goto %L_1
    // CHECK: }
    }

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.assign_label [[L]]
      clift.assign_label %L_0
      // CHECK-NOT clift.assign_label
      clift.assign_label %L_1
    // CHECK: }
    }
  // CHECK: }
  }
// CHECK: }
}
