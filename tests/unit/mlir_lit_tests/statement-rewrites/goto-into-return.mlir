//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %revngcliftopt %s --goto-into-return | FileCheck %s

!void = !clift.primitive<void 0>
!int32_t = !clift.primitive<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK-NOT: clift.make_label
    %label = clift.make_label

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK-NOT: clift.goto
      // CHECK: clift.return
      clift.goto %label
    // CHECK: }
    }
    // CHECK-NOT: clift.assign_label
    clift.assign_label %label

  // CHECK: }
  }
// CHECK: }
}
