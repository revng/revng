//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --hoist-terminal-branches | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>

!abort = !clift.func<"abort" : !void()>
!f = !clift.func<"f" : !void(!int32_t)>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  clift.func @abort<!abort>() -> !void attributes { noreturn } {}

  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>(%arg0 : !int32_t) -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.assign_label %0
    clift.assign_label %0
    // CHECK: clift.if {
    clift.if {
      // CHECK: %1 = clift.test %arg0 : !int32_t
      %1 = clift.test %arg0 : !int32_t
      // CHECK: clift.yield %1 : !clift.bool
      clift.yield %1 : !clift.bool
    // CHECK: } then {
    } then {
      // CHECK: clift.goto %0
      clift.goto %0
    // CHECK: }
    // CHECK-NOT: else
    } else {
      clift.expr {
        %1 = clift.use @abort : !abort
        %2 = clift.call %1() : !abort
        clift.yield %2 : !void
      }
    }
    // CHECK: clift.expr {
      // CHECK: %1 = clift.use @abort : !abort
      // CHECK: %2 = clift.call %1() : !abort
      // CHECK: clift.yield %2 : !void
    // CHECK: }
  // CHECK: }
  }
// CHECK: }
}
