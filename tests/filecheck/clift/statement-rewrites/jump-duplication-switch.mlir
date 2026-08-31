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
    // CHECK: %1 = clift.make_label
    %1 = clift.make_label
    // CHECK: clift.assign_label %0
    clift.assign_label %0
    // CHECK: clift.expr {
    clift.expr {
      // CHECK: %2 = clift.undef : !int32_t
      %2 = clift.undef : !int32_t
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: }
    }
    // CHECK: clift.assign_label %1
    clift.assign_label %1
    // CHECK: clift.switch {
    clift.switch {
      // CHECK: %2 = clift.imm 0 : !int32_t
      %2 = clift.imm 0 : !int32_t
      // CHECK: clift.yield %2 : !int32_t
      clift.yield %2 : !int32_t
    // CHECK: } case 0 {
    } case 0 {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %2 = clift.imm 1 : !int32_t
        %2 = clift.imm 1 : !int32_t
        // CHECK: clift.yield %2 : !int32_t
        clift.yield %2 : !int32_t
      // CHECK: }
      }
      clift.goto %0
    // CHECK: } case 1 {
    } case 1 {
      // CHECK: clift.expr {
      clift.expr {
        // CHECK: %2 = clift.imm 2 : !int32_t
        %2 = clift.imm 2 : !int32_t
        // CHECK: clift.yield %2 : !int32_t
        clift.yield %2 : !int32_t
      // CHECK: }
      }
      // CHECK: clift.goto %1
    // CHECK: } default {
      // CHECK: clift.goto %1
    // CHECK: }
    }
    // CHECK-NOT: clift.goto %1
    clift.goto %1
  // CHECK: }
  }
// CHECK: }
}
