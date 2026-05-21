//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-statements="enable-patterns=trivial-jump-elimination" | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

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
    // CHECK: } then {
    } then {
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
  // CHECK: }
  }
// CHECK: }
}
