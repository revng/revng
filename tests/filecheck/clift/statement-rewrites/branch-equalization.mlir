//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// RUN: %root/bin/revng clift-opt %s --optimize-statements | FileCheck %s

!void = !clift.void
!int32_t = !clift.int<signed 4>
!f = !clift.func<"/model-type/1001" : !void()>

// CHECK: module attributes {clift.module} {
module attributes {clift.module} {
  // CHECK: clift.func
  // CHECK-SAME: {
  clift.func @f<!f>() -> !void {
    // CHECK: %0 = clift.make_label
    %0 = clift.make_label
    // CHECK: clift.for body {
    clift.for body {
      // CHECK: clift.if {
      clift.if {
        %1 = clift.imm 1 : !int32_t
        clift.yield %1 : !int32_t
      // CHECK: } then {
      } then {
        // CHECK: clift.goto %0
        clift.goto %0
      // CHECK: } else {
      // CHECK: clift.goto %0
      // CHECK: }
      }
      // CHECK-not: clift.goto
      clift.goto %0
    // CHECK: }
    }
    // CHECK: clift.assign_label %0
    clift.assign_label %0
  // CHECK: }
  }
// CHECK: }
}
