//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --convert-void-return-to-goto | FileCheck %s

!void = !clift.void
!f = !clift.func<"/model-type/1001" : !void()>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.make_label
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.true
      %0 = clift.true
      // CHECK: clift.yield %1 : !clift.bool
      clift.yield %0 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK-NOT: clift.return
      // CHECK: clift.goto %0
      clift.return {}
    // CHECK: }
    }
    // CHECK: clift.assign_label %0
  // CHECK: }
  }
// CHECK: }
}
