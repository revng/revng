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
    %label_1 = clift.make_label
    %label_2 = clift.make_label
    %label_3 = clift.make_label

    // CHECK: clift.if {
    clift.if {
      clift.yield %arg0 : !int32_t
    // CHECK: } {
    } {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.goto
      clift.goto %label_1
    // CHECK: }
    }
    clift.assign_label %label_1

    // CHECK: clift.switch {
    clift.switch {
      clift.yield %arg0 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.goto
      clift.goto %label_3
    // CHECK: }
    } default {
      // CHECK: clift.expr
      clift.expr {
        clift.yield %arg0 : !int32_t
      // CHECK: }
      }

      // CHECK-NOT: clift.goto
      clift.goto %label_3
    }
    clift.assign_label %label_3

// WIP: Not yet implemented
// COM:    clift.assign_label %label_2
// COM:    // CHECK: clift.while {
// COM:    clift.while {
// COM:      clift.yield %arg0 : !int32_t
// COM:    // CHECK: } {
// COM:    } {
// COM:      // CHECK: clift.expr
// COM:      clift.expr {
// COM:        clift.yield %arg0 : !int32_t
// COM:      // CHECK: }
// COM:      }
// COM:
// COM:      // CHECK-NOT: clift.goto
// COM:      clift.goto %label_2
// COM:    // CHECK: }
// COM:    }

  // CHECK: }
  }
// CHECK: }
}
