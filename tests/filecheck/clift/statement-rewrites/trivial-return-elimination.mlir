//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --convert-void-return-to-goto --optimize-statements=enable-patterns=trivial-jump-elimination --convert-jump-to-void-return | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!f = !clift.func<"/model-type/1001" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: clift.if {
    clift.if {
      // CHECK: [[COND:%[0-9]+]] = clift.test %arg0 : !int32_t
      %1 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield [[COND]] : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK-NOT: clift.return
      clift.return {}
    // CHECK: }
    }
    // CHECK-NOT: clift.return
    clift.return {}
  // CHECK: }
  }
// CHECK: }
}
