//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --return-into-goto | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: [[L:%[0-9]+]] = clift.make_label

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK-NOT: clift.return
      // CHECK: clift.goto [[L]]
      clift.return {}
    // CHECK: }
    }
    // CHECK: clift.assign_label [[L]]

  // CHECK: }
  }
// CHECK: }
}
